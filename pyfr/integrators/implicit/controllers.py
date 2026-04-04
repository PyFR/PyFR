import numpy as np

from pyfr.integrators.base import StepInfo
from pyfr.integrators.controllers import CFLControllerMixin, PIControllerMixin
from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.integrators.implicit.newton import NewtonDivergenceError
from pyfr.nputil import LogGPOptimiser


class ThroughputLimitMixin:
    _tput_gp_wsize = 20
    _tput_fac_lo, _tput_fac_hi = 0.6, 1.67
    _tput_degrade_thresh = 0.7
    _tput_degrade_windows = 3
    _tput_grace_windows = 2

    def _init_tput_limit(self, sect, initsoln):
        self._tput_limit = self.cfg.getbool(sect, 'tput-limit', True)
        self.dt_update_interval = self.cfg.getint(sect, 'dt-update-interval',
                                                  100)
        self._growth_fact = self.cfg.getfloat(sect, 'growth-fact', 1.2)

        self._wtime_window = np.empty(self.dt_update_interval)
        self._fac_buffer = np.empty(self.dt_update_interval)
        self._dt_gp = LogGPOptimiser(self._tput_gp_wsize, (1, 1))
        self._steps_in_window = 0
        self._settled = False
        self._settled_tput = 0.0
        self._degrade_count = 0
        self._grace_count = 0
        self._krylov_was_settled = False
        self._explore_targets = []
        self._expand_cooldown = 0

    def _reset_tput(self, dt):
        lo = max(dt / 10**1.5, self.dtmin)
        hi = min(dt*10**1.5, self.dtmax)
        self._dt_gp.reset((lo, hi))
        self._settled = False
        self._settled_tput = 0.0
        self._degrade_count = 0
        self._grace_count = 0
        self._steps_in_window = 0
        self._expand_cooldown = 0

        # Bidirectional exploration: grow first, then shrink from initial
        g = self._growth_fact
        self._explore_targets = [
            dt,
            min(dt*g, self.dtmax),
            min(dt*g**2, self.dtmax),
            max(dt/g, self.dtmin),
            max(dt/g**2, self.dtmin),
        ]

    def _check_expand_bounds(self, best_dt):
        if not self._settled:
            return

        if self._expand_cooldown > 0:
            self._expand_cooldown -= 1
            return

        gp = self._dt_gp
        lo, hi = gp.x_lo, gp.x_hi
        rng = hi - lo
        margin = rng / 50
        step = np.log(self._growth_fact**2)
        lo_lim, hi_lim = np.log(self.dtmin), np.log(self.dtmax)
        best_log = np.log(best_dt)

        if best_log >= hi - margin and hi < hi_lim:
            gp.x_hi = min(hi + step, hi_lim)
            self._expand_cooldown = 5
        elif best_log <= lo + margin and lo > lo_lim:
            gp.x_lo = max(lo - step, lo_lim)
            self._expand_cooldown = 5

    def _throughput_limit(self, dt, wtime, fac):
        if not self._tput_limit:
            return fac

        # Wait for the Krylov tolerance controller, if any, to settle
        if not self._tol_controller.settled:
            self._steps_in_window = 0
            self._krylov_was_settled = False
            return min(1.0, fac)

        # First window after settling; reset throughput exploration
        if not self._krylov_was_settled:
            self._krylov_was_settled = True
            self._reset_tput(dt)

        i = self._steps_in_window
        self._wtime_window[i] = wtime
        self._fac_buffer[i] = fac
        self._steps_in_window += 1

        if self._steps_in_window < self.dt_update_interval:
            return min(1.0, fac)

        # Window complete; compute statistics
        med_tput = dt / np.mean(self._wtime_window)
        med_fac = np.exp(np.median(np.log(self._fac_buffer)))
        self._steps_in_window = 0

        return self._update_tput(dt, med_tput, med_fac)

    def _update_tput(self, dt, med_tput, med_fac):
        # Update the GP model with the new throughput measurement
        self._dt_gp.record(dt, med_tput)

        g = self._growth_fact
        n_explore = len(self._explore_targets)

        # Exploration phase; follow bidirectional targets
        if self._dt_gp.n < n_explore:
            return self._explore_targets[self._dt_gp.n] / dt

        # Exploit; find best dt
        best_dt = self._dt_gp.optimum(minimise=False)
        self._check_expand_bounds(best_dt)
        fac = np.clip(best_dt / dt, self._tput_fac_lo, self._tput_fac_hi)

        # Settle once the GP-optimal dt is close to the current dt
        if not self._settled and 1/g < fac < g:
            self._settled = True
            self._settled_tput = med_tput
            self._grace_count = self._tput_grace_windows

            # If dt has drifted from where Krylov built its model,
            # re-evaluate the tolerance at the new operating dt
            initial_dt = self._explore_targets[0]
            if abs(np.log10(dt / initial_dt)) > 0.1:
                self._tol_controller.soft_reset()

        # Still exploring; only constrain if dt controller wants to shrink
        if not self._settled:
            fac = min(fac, med_fac) if med_fac < 1.0 else fac
        # Grace period after settling; track baseline throughput
        elif self._grace_count > 0:
            self._grace_count -= 1
            self._settled_tput = med_tput
            fac = med_fac
        # Throughput healthy; no action required
        elif med_tput > self._tput_degrade_thresh*self._settled_tput:
            self._settled_tput = max(self._settled_tput, med_tput)
            self._degrade_count = 0
            fac = med_fac
        # Throughput degraded; re-explore after consecutive bad windows
        else:
            self._degrade_count += 1
            if self._degrade_count < self._tput_degrade_windows:
                fac = med_fac
            else:
                self._settled = False
                self._degrade_count = 0
                fac = min(fac, med_fac) if med_fac < 1.0 else fac

        return fac


class BaseImplicitController(BaseImplicitIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Stats on the most recent step
        self._stage_stats = []
        self.stepinfo = []

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()

    def _accept_step(self, dt, idxcurr, wtime, err=None):
        self._advance_time(dt)
        self.nacptsteps += 1

        self.stepinfo.append(StepInfo(dt, 'accept', err, wtime,
                                      self._stage_stats))
        self._stage_stats = []

        self.idxcurr = idxcurr

        self._invalidate_caches()

        # Run any plugins
        self._run_plugins()

        # Clear step info after plugins have consumed it
        self.stepinfo = []

    def _reject_step(self, dt, idxold, wtime, err=None):
        if dt <= self.dtmin:
            raise RuntimeError('Minimum sized time step rejected')

        self.nrjctsteps += 1

        self.stepinfo.append(StepInfo(dt, 'reject', err, wtime,
                                      self._stage_stats))
        self._stage_stats = []

        self.idxcurr = idxold


class ImplicitNoneController(BaseImplicitController):
    controller_name = 'none'
    controller_has_variable_dt = False
    controller_needs_errest = False
    controller_needs_cfl = False

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            # Decide on the time step
            dt = self._clamp_dt(self.dt, t)

            # Take the step
            idxcurr, wtime = self._timed_step(self.tcurr, dt)

            # We are not adaptive, so accept every step
            self._accept_step(dt, idxcurr, wtime)


class ImplicitPIController(ThroughputLimitMixin, PIControllerMixin,
                           BaseImplicitController):
    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        self._init_pi_controller()

        sect = 'solver-time-integrator'
        self._pi_alpha = self.cfg.getfloat(sect, 'pi-alpha', 0.7)
        self._pi_beta = self.cfg.getfloat(sect, 'pi-beta', 0.4)

        # Initialise dt/errprev from restart or defaults
        if initsoln and (sd := initsoln.state.get('intg/ctrl')):
            self.dt, self._errprev = sd[:2]
        else:
            self._errprev = 1.0

        self.serialiser.register('intg/ctrl',
                                 lambda: [self.dt, self._errprev])

        self._init_tput_limit(sect, initsoln)

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        expa = self._pi_alpha / self.stepper_order
        expb = self._pi_beta / self.stepper_order

        while self.tcurr < t:
            # Decide on the time step
            dt = self._clamp_dt(min(self.dt, self.dtmax), t)

            # Take the step
            (icurr, iprev, ierr), wtime = self._timed_step(self.tcurr, dt)

            # Estimate the error
            err = self._errest(icurr, ierr)

            # Determine the time step adjustment factor
            fac = err**-expa*self._errprev**expb
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
    pass


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

        self._nfailures = 0
        self._failure_fact = cfg.getfloat(sect, 'failure-fact', 0.5)
        self._max_failures = cfg.getint(sect, 'max-failures', 5)

        self._init_dt(initsoln)
        self._init_tput_limit(sect, initsoln)

    def _init_dt(self, initsoln):
        sd = initsoln.state.get('intg/ctrl') if initsoln else None

        if sd is not None:
            self.dt = sd[0]

        self.serialiser.register('intg/ctrl', lambda: [self.dt])

    def _adapt_dt(self, dt, wtime):
        fac = self._throughput_limit(dt, wtime, 1.0)

        if fac != 1.0:
            self.dt = max(min(fac*self.dt, self.dtmax), self.dtmin)

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            # Decide on the time step
            dt = self._clamp_dt(min(self.dt, self.dtmax), t)

            try:
                # Take the step
                idxcurr, wtime = self._timed_step(self.tcurr, dt)

                # Adapt dt based on throughput
                self._adapt_dt(dt, wtime)

                self._nfailures = 0
                self._accept_step(dt, idxcurr, wtime)
            except NewtonDivergenceError:
                self._nfailures += 1

                if self._nfailures > self._max_failures:
                    raise NewtonDivergenceError(
                        f'Failed {self._nfailures} times consecutively at '
                        f'dt={dt:.2e}'
                    )

                dt = self._failure_fact*dt
                if dt < self.dtmin:
                    raise RuntimeError(f'dt={dt:.2e} below minimum '
                                       f'{self.dtmin:.2e}')

                self.dt = dt
                self._reject_step(dt, self.idxcurr, 0.0)
