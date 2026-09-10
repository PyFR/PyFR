import numpy as np

from pyfr.integrators.base import StepInfo
from pyfr.integrators.controllers import CFLControllerMixin, PIControllerMixin
from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.integrators.implicit.nonlinear import NonlinearDivergenceError
from pyfr.nputil import LogGPOptimiser


class PrecondDriftMonitor:
    # Windows to average when establishing a baseline
    _baseline_windows = 2

    # Minimum improvement ratio to consider a refresh successful
    _refresh_improvement = 0.9

    # Consecutive failed refreshes before disabling
    _max_refresh_failures = 3

    # (min gdt mismatch, drift threshold, consecutive windows)
    _drift_tiers = [(0.3, 1.5, 1), (0.1, 1.5, 2), (-1, 2.0, 3)]

    def __init__(self):
        self.reset()
        self._refresh_failures = 0
        self._refresh_disabled = False
        self._prev_baseline = None

    def reset(self):
        self._baseline = None
        self._baseline_buf = []
        self._drift_count = 0
        self._krylov_total = 0
        self._newton_total = 0

    def record(self, nkrylov, niters):
        self._krylov_total += nkrylov
        self._newton_total += niters

    def check(self, dt, gamma, gdt_built, invalidate_fn):
        # No iteration data accumulated this window
        if self._newton_total == 0:
            return

        # Mean Krylov iterations per Newton step for this window
        mean_k = self._krylov_total / self._newton_total
        self._krylov_total = self._newton_total = 0

        # Establish a baseline before checking for drift
        if self._baseline is None:
            self._update_baseline(mean_k)
        else:
            self._detect_drift(mean_k, dt, gamma, gdt_built, invalidate_fn)

    def _update_baseline(self, mean_k):
        # Accumulate windows until we have enough to average
        self._baseline_buf.append(mean_k)
        if len(self._baseline_buf) < self._baseline_windows:
            return

        # Set the baseline from the accumulated windows
        self._baseline = np.mean(self._baseline_buf)
        self._baseline_buf = []

        # If this baseline follows a refresh, check if it helped
        if self._prev_baseline is not None:
            if self._baseline < self._refresh_improvement*self._prev_baseline:
                self._refresh_failures = 0
            else:
                self._refresh_failures += 1
                if self._refresh_failures >= self._max_refresh_failures:
                    self._refresh_disabled = True
            self._prev_baseline = None

    def _detect_drift(self, mean_k, dt, gamma, gdt_built, invalidate_fn):
        # Gamma*dt mismatch relative to the dt at last build
        gdt_mismatch = abs(np.log10(gamma*dt / gdt_built))

        # Tiered response: larger mismatch => act sooner
        thresh, nwindows = next(
            (t, n) for lo, t, n in self._drift_tiers if gdt_mismatch > lo
        )

        # Count consecutive windows with Krylov drift above the threshold
        if mean_k / self._baseline > thresh:
            self._drift_count += 1
        else:
            self._drift_count = 0

        # Trigger a rebuild once the drift persists long enough
        if self._drift_count >= nwindows and not self._refresh_disabled:
            self._prev_baseline = self._baseline
            invalidate_fn()
            self.reset()


class ThroughputLimitMixin:
    # GP optimiser sliding window size
    _tput_gp_wsize = 20

    # dt adjustment factor bounds
    _tput_fac_lo, _tput_fac_hi = 0.6, 1.67

    # Throughput fraction below settled to count as degraded
    _tput_degrade_thresh = 0.7

    # Consecutive degraded windows before re-exploring
    _tput_degrade_windows = 3

    # Grace windows after settling to establish throughput baseline
    _tput_grace_windows = 2

    def _init_tput_limit(self, initsoln):
        sect = 'solver-time-integrator'

        self._tput_limit = self.cfg.getbool(sect, 'tput-limit', True)
        self.dt_update_interval = self.cfg.getint(sect, 'dt-update-interval',
                                                  100)
        self._growth_fact = self.cfg.getfloat(sect, 'growth-fact', 1.2)

        self._wtime_window = np.empty(self.dt_update_interval)
        self._fac_buffer = np.empty(self.dt_update_interval)
        self._dt_gp = LogGPOptimiser(self._tput_gp_wsize, (1, 1))
        self._krylov_was_settled = False
        self._explore_targets = []
        self._pc_monitor = PrecondDriftMonitor()
        self._zero_window_state()

        # Free-solve gate: ignore windows where the linear solve is trivial
        self._tput_kpn_min = 1.0

        # Periodic dt re-exploration with exponential backoff (in windows)
        self._reexplore_base = self.cfg.getint(sect, 'dt-reexplore-interval',
                                               50)
        self._reexplore_max = 3200
        self._reexplore_interval = self._reexplore_base
        self._reexplore_countdown = self._reexplore_base
        self._reexplore_dt = None

    def _zero_window_state(self):
        self._settled = False
        self._settled_tput = 0.0
        self._degrade_count = 0
        self._grace_count = 0
        self._steps_in_window = 0
        self._expand_cooldown = 0
        self._win_krylov = 0
        self._win_newton = 0

    def _reset_tput(self, dt):
        lo = max(dt / 10**1.5, self.dtmin)
        hi = min(dt*10**1.5, self.dtmax)
        self._dt_gp.reset((lo, hi))
        self._zero_window_state()

        # Bidirectional exploration: grow first, then shrink from initial
        g = self._growth_fact
        facs = (1, g, g*g, 1/g, 1/(g*g))
        self._explore_targets = [np.clip(dt*f, self.dtmin, self.dtmax)
                                 for f in facs]

    def _check_expand_bounds(self, best_dt):
        if not self._settled:
            return

        if self._expand_cooldown > 0:
            self._expand_cooldown -= 1
            return

        # Current GP search bounds with a margin to detect edge hits
        gp = self._dt_gp
        lo, hi = gp.x_lo, gp.x_hi
        margin = (hi - lo) / 50

        # Expansion step and absolute dt limits
        step = np.log(self._growth_fact**2)
        lo_lim, hi_lim = np.log(self.dtmin), np.log(self.dtmax)
        best_log = np.log(best_dt)

        # Expand the search range if the optimum is at the edge
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

        # Accumulate iteration data from stages at the settled tolerance
        best_tol = self._tol_controller.best_tol
        for s in self._stage_stats:
            if abs(np.log10(s.inner_tol / best_tol)) <= 0.15:
                self._pc_monitor.record(s.nmatvec, s.niters)

        # Check if the preconditioner was rebuilt during this step
        step_had_build = any(s.precond_built for s in self._stage_stats)

        # Exclude build steps so the rebuild cost does not pollute the GP
        if step_had_build:
            if self._steps_in_window < self.dt_update_interval:
                return min(1.0, fac)
        else:
            i = self._steps_in_window
            self._wtime_window[i] = wtime
            self._fac_buffer[i] = fac
            self._steps_in_window += 1
            self._win_krylov += sum(s.nmatvec for s in self._stage_stats)
            self._win_newton += sum(s.niters for s in self._stage_stats)

        if self._steps_in_window < self.dt_update_interval:
            return min(1.0, fac)

        n = self._steps_in_window
        med_tput = dt / np.mean(self._wtime_window[:n])
        med_fac = np.exp(np.median(np.log(self._fac_buffer[:n])))
        self._steps_in_window = 0

        # Mean Krylov iterations per Newton iteration over the window
        kpn = self._win_krylov / max(self._win_newton, 1)
        self._win_krylov = 0
        self._win_newton = 0

        if self._preconditioner.active:
            self._pc_monitor.check(dt, self._gamma, self._precond_gdt_built,
                                   self._invalidate_precond)

        # Trivial solve => dt below stiff regime; grow, do not record it
        if kpn < self._tput_kpn_min:
            return self._growth_fact

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
            self._on_settle(dt, med_tput)

        # Resolve the step factor from the post-settle state machine
        return self._resolve_tput_fac(dt, med_tput, med_fac, fac)

    def _on_settle(self, dt, med_tput):
        self._settled = True
        self._settled_tput = med_tput
        self._grace_count = self._tput_grace_windows

        # Re-evaluate tolerance if dt drifted from where Krylov built its model
        initial_dt = self._explore_targets[0]
        if abs(np.log10(dt / initial_dt)) > 0.1:
            self._tol_controller.soft_reset()

        # Backoff: grow interval if re-explored dt confirmed, else reset
        if self._reexplore_dt is not None:
            if abs(np.log10(dt / self._reexplore_dt)) < 0.1:
                nxt = 2*self._reexplore_interval
                self._reexplore_interval = min(nxt, self._reexplore_max)
            else:
                self._reexplore_interval = self._reexplore_base
            self._reexplore_dt = None
        self._reexplore_countdown = self._reexplore_interval

        # On a long freeze, rebuild precond if built at a different dt
        gdt_built = self._precond_gdt_built
        long_freeze = self._reexplore_interval > self._reexplore_base
        if self._preconditioner.active and gdt_built > 0 and long_freeze:
            gdt = self._gamma*dt
            if abs(np.log10(gdt / gdt_built)) > 0.1:
                self._invalidate_precond()

    def _resolve_tput_fac(self, dt, med_tput, med_fac, fac):
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

            # Periodically re-explore dt even when healthy (exp. backoff)
            self._reexplore_countdown -= 1
            if self._reexplore_countdown <= 0:
                self._reexplore_dt = dt
                self._reset_tput(dt)

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

        while self.tcurr < t and self.tcurr < self.tend:
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

        self._init_pi_controller(initsoln)
        self._init_tput_limit(initsoln)

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        expa = self._pi_alpha / self.stepper_order
        expb = self._pi_beta / self.stepper_order

        while self.tcurr < t and self.tcurr < self.tend:
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

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t and self.tcurr < self.tend:
            # Decide on the time step
            dt = self._clamp_dt(min(self.dt, self.dtmax), t)

            try:
                # Take the step
                idxcurr, wtime = self._timed_step(self.tcurr, dt)

                # Adapt dt based on throughput
                self._adapt_dt(dt, wtime)

                self._nfailures = 0
                self._accept_step(dt, idxcurr, wtime)
            except NonlinearDivergenceError:
                # Force a preconditioner rebuild
                self._invalidate_precond()
                self._nfailures += 1

                # Bail if we have exceeded the failure limit
                if self._nfailures > self._max_failures:
                    raise NonlinearDivergenceError(
                        f'Failed {self._nfailures} times consecutively at '
                        f'dt={dt:.2e}'
                    )

                # Reduce dt and retry
                dt = self._failure_fact*dt
                if dt < self.dtmin:
                    raise RuntimeError(f'dt={dt:.2e} below minimum '
                                       f'{self.dtmin:.2e}')

                self.dt = dt
                self._reject_step(dt, self.idxcurr, 0.0)
