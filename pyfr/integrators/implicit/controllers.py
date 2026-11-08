import time
from statistics import NormalDist

import numpy as np

from pyfr.integrators.base import StepInfo
from pyfr.integrators.controllers import CFLControllerMixin, PIControllerMixin
from pyfr.integrators.implicit.base import (BaseImplicitIntegrator,
                                            DivergenceError)
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
from pyfr.nputil import LogGPOptimiser


class PrecondDriftMonitor:
    def __init__(self, interval):
        self._interval = interval
        self._key = None

    def _reset(self):
        self._baseline = None
        self._excess = 0.0
        self._napps = 0
        self._wtime = 0.0
        self._nsteps = 0

    def record(self, key, napps, wtime, build_wtime):
        # Start afresh when the preconditioner or Krylov tolerance changes
        if key != self._key:
            self._key = key
            self._reset()
            return False

        # Take the first window after a build as the baseline
        if self._baseline is None:
            # Accumulate the preconditioner applications and wall time
            self._napps += napps
            self._wtime += wtime
            self._nsteps += 1

            # Mean preconditioner applications per step
            if self._nsteps >= self._interval and self._napps:
                self._baseline = self._napps / self._nsteps

            return False
        # Otherwise accumulate the wall time spent on excess applications
        elif napps:
            excess = self._excess + (1 - self._baseline / napps)*wtime
            self._excess = max(excess, 0.0)

            # Rebuild once the drift has cost more wall time than a build
            return self._excess > build_wtime
        else:
            return False


class ThroughputLimitMixin:
    # GP optimiser sliding window size
    _tput_gp_wsize = 20

    # Clean windows before the post-failure dt ceiling starts to lift
    _tput_hold_windows = 2

    # Throughput fraction within which two time steps are tied
    _tput_tie_tol = 0.05

    # Retries at one dt within a window before dt itself is blamed
    _tput_max_retries = 2

    # Probability of a sound model being discarded as shifted
    _tput_shift_delta = 0.1

    # Windows the model must hold before it can be contradicted
    _tput_shift_nmin = 3

    # Scale a median absolute deviation to a normal standard deviation
    _mad_to_sigma = 1/NormalDist().inv_cdf(3/4)

    # Floor on the robust wall-time scale, relative to the window median
    _tput_scale_floor = 1e-3

    def _init_tput_limit(self, initsoln):
        sect = 'solver-time-integrator'

        self._tput_limit = self.cfg.getbool(sect, 'tput-limit', True)
        self.dt_update_interval = self.cfg.getint(sect, 'dt-update-interval',
                                                  100)
        self._growth_fact = self.cfg.getfloat(sect, 'growth-fact', 1.2)

        # Size Krylov tolerance probes to fit within a window of steps
        if self._tput_limit:
            nsolves = self.dt_update_interval*self.nimplicit
            self._tol_controller.fit_window(nsolves)

        self._wtime_window = np.empty(self.dt_update_interval)
        self._fac_buffer = np.empty(self.dt_update_interval)
        # Windows over which a throughput observation is forgotten
        decay = self.cfg.getint(sect, 'dt-reexplore-interval', 50)

        # Windows a dt is held before a neighbouring dt is re-measured
        self._reexplore_windows = max(decay // 2, 1)
        self._hold_windows = 0

        # Log throughput difference within which two time steps are tied
        self._ltie = np.log(1 + self._tput_tie_tol)

        # Model log throughput against log dt
        self._dt_gp = LogGPOptimiser(self._tput_gp_wsize, (1, 1), trend=True,
                                     decay=decay,
                                     ls=2*np.log(self._growth_fact))
        self._tput_started = False
        self._reset_window()

        # Cap on dt after a failed step, lifted over clean windows
        self._dt_ceiling = float('inf')
        self._ceiling_hold = self._ceiling_steps = 0

        # Largest dt and total wall time of the failed attempts at a step
        self._fail_dt, self._fail_wtime = 0.0, 0.0

        # Retries at one dt since the last completed window
        self._retry_dt, self._nretries = 0.0, 0

        # Whether the next preconditioner build was forced by a failure
        self._charge_build = False

        # Windows recorded since the model was last started afresh
        self._shift_nwin = 0

        # Set once a window of solves has needed more than one iteration
        self._tput_stiff = False

        # Cost of the incumbent Krylov tolerance the model was measured at
        self._tput_tol_cost = 0.0

    def _reset_window(self):
        self._steps_in_window = 0
        self._win_iters = self._win_stages = 0

    def _reset_tput(self):
        self._reset_model()
        self._tput_stiff = False
        self._reset_window()

    def _record_failed_attempt(self, dt, wtime, retry):
        # Count the failed attempt towards the step which eventually succeeds
        self._fail_dt = max(self._fail_dt, dt)
        self._fail_wtime += self._solve_wtime(wtime)

        # Charge the rebuild forced by the previous failure to this step too
        if self._charge_build:
            self._fail_wtime += wtime - self._solve_wtime(wtime)
        self._charge_build = True

        # A retry blames the preconditioner; count retries at this dt
        if retry:
            self._nretries = self._nretries + 1 if dt == self._retry_dt else 1
            self._retry_dt = dt

            # Only repeated retries at one dt within a window implicate dt
            if self._nretries < self._tput_max_retries:
                return

        # Cap dt one growth step below the failure until windows run clean
        self._dt_ceiling = min(self._dt_ceiling, dt / self._growth_fact)
        self._ceiling_hold = self._tput_hold_windows
        self._ceiling_steps = 0

    def _solve_wtime(self, wtime):
        # Wall time of the step less that of any preconditioner build
        if any(s.precond_built for s in self._stage_stats):
            return max(wtime - self._preconditioner.build_wtime, 0.0)
        else:
            return wtime

    def _lift_ceiling(self):
        if self._ceiling_hold > 0:
            self._ceiling_hold -= 1
        else:
            self._dt_ceiling = min(self._dt_ceiling*self._growth_fact,
                                   self.dtmax)

    def _explore_fac(self, best_dt, dt):
        g = self._growth_fact
        lo, hi = self._dt_gp.x_range

        # Climb while the top of the data is not significantly worse
        hi_lim = min(self.dtmax, self._dt_ceiling)
        if self._tput_tied(best_dt, hi):
            # Step beyond the top of the data from the top itself
            if dt > hi / g**0.5:
                target = hi*g
            # Re-measure the top only once its observation has gone stale
            elif self._dt_gp.age_at(hi) >= self._reexplore_windows:
                target = hi
            else:
                return None

            if target <= hi_lim:
                return min(target / dt, g)
            else:
                return None
        # Descend only when the bottom of the data beats the level above it
        elif best_dt < lo*g**0.5 and lo/g >= self.dtmin:
            if self._tput_tied(lo, lo*g):
                return None
            else:
                return max(lo / (g*dt), 1/g)
        else:
            return None

    def _tput_tied(self, best_dt, dt):
        # Count dt as tied with the best unless it is significantly worse
        mu_best = self._dt_gp.mean_at(best_dt)
        mu = self._dt_gp.mean_at(dt)
        return mu_best - mu < self._ltie

    def _hold_fac(self, dt):
        # Hold until a re-explore interval of windows has run at this dt
        self._hold_windows += 1
        if self._hold_windows < self._reexplore_windows:
            return 1.0

        # Re-measure whichever neighbouring dt was measured longest ago
        g = self._growth_fact
        hi_lim = min(self.dtmax, self._dt_ceiling)
        cands = [c for c in (dt*g, dt/g) if self.dtmin <= c <= hi_lim]
        if cands:
            target = max(cands, key=self._dt_gp.age_at)
            return target / dt
        else:
            return 1.0

    def _throughput_limit(self, dt, wtime, fac):
        # Charge the rebuild forced by a failed attempt to that attempt
        if self._charge_build:
            self._fail_wtime += wtime - self._solve_wtime(wtime)
            self._charge_build = False

        fac = self._window_fac(dt, wtime, fac)

        # Record the failed attempts as a poor throughput at the failed dt
        if self._fail_dt and self._tput_started:
            tput = dt / (self._fail_wtime + self._solve_wtime(wtime))

            # A failure the model did not foresee is a shift like any other
            if self._tput_shifted(self._fail_dt, np.log(tput), self._ltie**2):
                self._reset_model()

            self._dt_gp.record(self._fail_dt, np.log(tput))

        self._fail_dt, self._fail_wtime = 0.0, 0.0

        # Release the ceiling after each window of steps without a failure
        self._ceiling_steps += 1
        if self._ceiling_steps >= self.dt_update_interval:
            self._ceiling_steps = 0
            self._lift_ceiling()

        # Cap growth at the post-failure ceiling
        return min(fac, self._dt_ceiling / dt)

    def _window_fac(self, dt, wtime, fac):
        if not self._tput_limit:
            return fac

        # Skip steps taken at a Krylov tolerance other than the incumbent
        if self._tol_controller.probing_other:
            return min(1.0, fac)

        # First window; reset throughput exploration
        if not self._tput_started:
            self._tput_started = True
            self._reset_tput()

        # Rescale the model by the cost change of a new incumbent tolerance
        tol_cost = self._tol_controller.relative_cost
        if tol_cost != self._tput_tol_cost:
            cfac = 10**(self._tput_tol_cost - tol_cost)
            self._dt_gp.shift(np.log(cfac))
            self._tput_tol_cost = tol_cost

        # Measure the step without the cost of any preconditioner build
        i = self._steps_in_window
        self._wtime_window[i] = self._solve_wtime(wtime)
        self._fac_buffer[i] = fac
        self._steps_in_window += 1
        self._win_iters += sum(s.nonlin_iters for s in self._stage_stats)
        self._win_stages += len(self._stage_stats)

        if self._steps_in_window < self.dt_update_interval:
            return min(1.0, fac)

        n = self._steps_in_window
        wtimes = self._wtime_window[:n]
        wmed = np.median(wtimes)
        med_tput = dt / wmed
        med_fac = np.exp(np.median(np.log(self._fac_buffer[:n])))
        self._steps_in_window = 0

        # A completed window clears the retry count at this dt
        self._nretries = 0

        # Sampling variance of the window throughput, robust to bursts
        mad = np.median(np.abs(wtimes - wmed))
        wsd = max(self._mad_to_sigma*mad, self._tput_scale_floor*wmed)
        tput_var = (np.pi / (2*n))*(wsd*med_tput / wmed)**2

        # Mean nonlinear iterations per stage over the window
        nps = self._win_iters / max(self._win_stages, 1)
        self._win_iters = self._win_stages = 0

        # Grow without recording while the solves are trivial
        trivial = nps <= 1 and self.requires_nonlinear_solver
        if not self._tput_stiff and trivial:
            return self._growth_fact
        else:
            self._tput_stiff = True

        return self._model_fac(dt, med_tput, tput_var, med_fac)

    def _tput_shifted(self, dt, ltput, lvar):
        comm, rank, root = get_comm_rank_root()

        # Too few windows for a prediction worth contradicting
        if self._shift_nwin < self._tput_shift_nmin:
            return False

        # Model prediction at this dt
        mu, var = self._dt_gp.predict(dt)

        # Confidence width over every observation the model holds
        rho = 2*np.log((np.pi*self._dt_gp.n)**2 / (3*self._tput_shift_delta))

        # A shift is a residual outside the model and noise confidence bands
        shifted = abs(ltput - mu) > np.sqrt(rho)*(np.sqrt(var) + np.sqrt(lvar))

        # Take the root's verdict so that every rank keeps the same model
        return comm.bcast(bool(shifted), root=root)

    def _reset_model(self):
        # Start the model afresh, keeping the stiff flag and the ceiling
        self._dt_gp.reset((self.dtmin, self.dtmax))
        self._hold_windows = 0
        self._shift_nwin = 0

    def _model_fac(self, dt, med_tput, tput_var, med_fac):
        comm, rank, root = get_comm_rank_root()
        lvar = max(tput_var / med_tput**2, self._ltie**2)

        # Discard a model which this window contradicts
        if self._tput_shifted(dt, np.log(med_tput), lvar):
            self._reset_model()

        # Update the GP model with the new throughput measurement
        self._dt_gp.record(dt, np.log(med_tput), lvar)
        self._shift_nwin += 1

        g = self._growth_fact

        # Exploit; find the best dt within the range of the observations
        best_dt = self._dt_gp.optimum(minimise=False, pad=0.0)

        # Too few observations to compare; climb
        if best_dt is None:
            fac = g
        # Keep exploring outwards while the optimum is not bracketed
        elif (efac := self._explore_fac(best_dt, dt)):
            fac = efac
        # Hold while the current dt is as good as the best
        elif self._tput_tied(best_dt, dt):
            fac = self._hold_fac(dt)
        else:
            fac = np.clip(best_dt / dt, 1/g, g)

        # Never grow a dt the error controller is shrinking
        if med_fac < 1.0:
            fac = min(fac, med_fac)

        # Take the root's factor so that every rank steps with the same dt
        fac = comm.bcast(float(fac), root=root)

        # A move ends the hold
        if fac != 1.0:
            self._hold_windows = 0

        return fac


class BaseImplicitController(BaseImplicitIntegrator):
    # Maximum time step; adaptive controllers read this from the config
    dtmax = float('inf')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = 'solver-time-integrator'

        # Stats on the most recent step
        self._stage_stats = []
        self.stepinfo = []

        # Response to nonlinear solver failures
        self._nfailures = 0
        self._failure_fact = self.cfg.getfloat(sect, 'failure-fact', 0.5)
        self._max_failures = self.cfg.getint(sect, 'max-failures', 5)

        # Preconditioner drift monitoring
        interval = self.cfg.getint(sect, 'precond-check-interval', 50)
        self._pc_monitor = PrecondDriftMonitor(interval)

        # Start the plugins and, if not restarting, fire off any event handlers
        self._start_plugins()

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t and self.tcurr < self.tend:
            self._step(self._next_dt(t))

    def _next_dt(self, t):
        return self._clamp_dt(min(self.dt, self.dtmax), t)

    def _step(self, dt):
        comm, _, _ = get_comm_rank_root()
        pc, backend = self._preconditioner, self.backend
        tstart = time.perf_counter() - backend.kernel_setup_wtime

        # Note which preconditioner build, if any, the step starts with
        build = pc.nbuilds if pc.active and pc.computed else None

        # Let the Krylov tolerance controller change tolerance for this step
        self._tol_controller.begin_step()

        try:
            ret = self._timed_step(self.tcurr, dt)
        except DivergenceError as err:
            # Agree on the time of the failed attempt across ranks
            wtime = time.perf_counter() - backend.kernel_setup_wtime - tstart
            wtime = scal_coll(comm.Allreduce, wtime, op=mpi.MAX)

            # Recover; retry as is only if the preconditioner predates the step
            self._recover_step(dt, wtime, pc.nbuilds == build, err)
        else:
            self._nfailures = 0
            self._finish_step(dt, *ret)

    def _record_failed_attempt(self, dt, wtime, retry):
        pass

    def _finish_step(self, dt, idxcurr, wtime):
        self._accept_step(dt, idxcurr, wtime)

    def _recover_step(self, dt, wtime, retry, err):
        # Bail if we have exceeded the failure limit
        self._nfailures += 1
        if self._nfailures > self._max_failures:
            msg = f'Failed {self._nfailures} consecutive times at {dt=:.2e}'
            raise DivergenceError(msg) from err

        # Rebuild the preconditioner
        self._invalidate_precond()

        # Retry as is, else reduce the time step if we are able to
        if retry:
            pass
        elif self.controller_has_variable_dt:
            self.dt = self._failure_fact*dt
            if self.dt < self.dtmin:
                msg = f'dt={self.dt:.2e} below minimum {self.dtmin:.2e}'
                raise RuntimeError(msg) from err
        else:
            msg = f'{err} at t={self.tcurr:.6g}, {dt=:.2e}'
            raise DivergenceError(msg) from err

        # Let the time step controller respond to the failure
        self._record_failed_attempt(dt, wtime, retry)

        # Record the failed attempt and how we are responding to it
        action = 'retry' if retry else 'fail'
        self._reject_step(dt, self.idxcurr, wtime, action=action)

    def _monitor_precond(self, wtime):
        pc, tc = self._preconditioner, self._tol_controller

        # Only monitor steps taken at the incumbent Krylov tolerance
        if not pc.active or tc.probing:
            return

        # Rebuild if the preconditioner work has drifted since the last build
        napps = sum(s.precond_apps for s in self._stage_stats)
        key = (pc.nbuilds, round(4*np.log10(tc.best_tol)))
        if self._pc_monitor.record(key, napps, wtime, pc.build_wtime):
            self._invalidate_precond()

    def _accept_step(self, dt, idxcurr, wtime, err=None):
        self._advance_time(dt)
        self.nacptsteps += 1

        # Check if the preconditioner needs refreshing
        self._monitor_precond(wtime)

        self.stepinfo.append(StepInfo(dt, 'accept', err, wtime,
                                      self._stage_stats))
        self._stage_stats = []

        self.idxcurr = idxcurr

        self._invalidate_caches()

        # Run any plugins
        self._run_plugins()

        # Clear step info after plugins have consumed it
        self.stepinfo = []

    def _reject_step(self, dt, idxold, wtime, err=None, action='reject'):
        if dt <= self.dtmin:
            raise RuntimeError('Minimum sized time step rejected')

        self.nrjctsteps += 1

        self.stepinfo.append(StepInfo(dt, action, err, wtime,
                                      self._stage_stats))
        self._stage_stats = []

        self.idxcurr = idxold


class ImplicitNoneController(BaseImplicitController):
    controller_name = 'none'
    controller_has_variable_dt = False
    controller_needs_errest = False
    controller_needs_cfl = False


class ImplicitPIController(ThroughputLimitMixin, PIControllerMixin,
                           BaseImplicitController):
    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        self._init_pi_controller(initsoln)
        self._init_tput_limit(initsoln)

        # Exponents for the error-based step size adjustment
        self._expa = self._pi_alpha / self.stepper_order
        self._expb = self._pi_beta / self.stepper_order

    def _finish_step(self, dt, idxs, wtime):
        icurr, iprev, ierr = idxs

        # Estimate the error
        err = self._errest(icurr, ierr)

        # Determine the time step adjustment factor
        fac = err**-self._expa*self._errprev**self._expb
        fac = min(self._maxfac, max(self._minfac, self._saffac*fac))

        # Apply throughput-based limiting
        fac = self._throughput_limit(dt, wtime, fac)

        # Compute the size of the next step
        self.dt = fac*dt

        # Decide if to accept or reject the step
        if err < 1.0:
            self._errprev = err
            self._accept_step(dt, icurr, wtime, err=err)
        else:
            self._reject_step(dt, iprev, wtime, err=err)


class ImplicitCFLController(CFLControllerMixin, BaseImplicitController):
    def _cfl_due(self):
        return not self._nfailures and super()._cfl_due()


class ImplicitThroughputController(ThroughputLimitMixin,
                                   BaseImplicitController):
    controller_name = 'throughput'
    controller_has_variable_dt = True
    controller_needs_errest = False
    controller_needs_cfl = False

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        sect = 'solver-time-integrator'

        self.dtmax = cfg.getfloat(sect, 'dt-max', 1e2)

        self._init_dt(initsoln)
        self._init_tput_limit(initsoln)

    def _init_dt(self, initsoln):
        sd = initsoln.state.get('intg/ctrl') if initsoln else None

        if sd is not None:
            self.dt = sd[0]

        self.serialiser.register('intg/ctrl', lambda: [self.dt])

    def _adapt_dt(self, dt, wtime):
        fac = self._throughput_limit(dt, wtime, 1.0)

        if fac != 1.0:
            self.dt = max(min(fac*self.dt, self.dtmax), self.dtmin)

    def _finish_step(self, dt, idxcurr, wtime):
        # Adapt dt based on throughput
        self._adapt_dt(dt, wtime)

        self._accept_step(dt, idxcurr, wtime)
