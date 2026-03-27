import numpy as np

from pyfr.nputil import LogGPOptimiser
from pyfr.util import subclass_where


def get_krylov_tol_controller(cfg, serialiser, initsoln):
    sect = 'solver-time-integrator'
    name = cfg.get(sect, 'krylov-tol-controller', 'none')
    cls = subclass_where(BaseToleranceController, name=name)

    return cls(cfg, serialiser, initsoln)


class BaseToleranceController:
    name = None
    max_retries = 0
    settled = True

    def update(self, wall_time, gamma_dt, success):
        pass

    def reset(self):
        pass

    def soft_reset(self):
        pass


class NullToleranceController(BaseToleranceController):
    name = 'none'

    def __init__(self, cfg, serialiser, initsoln):
        sect = 'solver-time-integrator'
        self._tol = cfg.getfloat(sect, 'krylov-rtol', 1e-2)

    def select_tolerance(self):
        return self._tol


class BaseProbeController(BaseToleranceController):
    def __init__(self, cfg, serialiser, initsoln):
        sect = 'solver-time-integrator'
        self._probe_nsolves = cfg.getint(sect, 'krylov-probe-nsolves', 20)

        self._ref_gdt = None
        self._steps_until_probe = 0

        # N-step probe window state
        self._probe_remaining = 0
        self._probe_costs = []
        self._probe_tol = None

    def _abort_probe(self):
        self._probe_remaining = 0
        self._probe_costs = []
        self._probe_tol = None

    def _start_probe(self, tol):
        self._probe_tol = tol
        self._probe_remaining = self._probe_nsolves
        self._probe_costs = []
        return tol

    def update(self, wall_time, gamma_dt, success):
        # Detect sudden dt jumps (Newton failure etc)
        if self._ref_gdt is not None and self._ref_gdt > 0:
            if abs(np.log10(gamma_dt / self._ref_gdt)) > 0.3:
                self._abort_probe()
                self.reset()

        self._ref_gdt = gamma_dt

        cost = np.log10(max(wall_time / gamma_dt, 1e-10)) + (not success)

        # Accumulate cost during probe window
        if self._probe_remaining > 0:
            self._probe_costs.append(cost)
            self._probe_remaining -= 1

            if self._probe_remaining == 0:
                self._finalize_probe()

        if not success:
            self._abort_probe()
            self._best_tol = self._rtol_max
            self._steps_until_probe = 0

    def select_tolerance(self):
        # Currently in a probe window; keep returning probe tolerance
        if self._probe_remaining > 0:
            return self._probe_tol
        # Waiting between probes; return incumbent
        elif self._steps_until_probe > 0:
            self._steps_until_probe -= 1
            return self._best_tol
        else:
            return self._next_probe()


class WindowedGPController(BaseProbeController):
    name = 'windowed-gp'
    max_retries = 2
    _base_interval = 100

    def __init__(self, cfg, serialiser, initsoln):
        super().__init__(cfg, serialiser, initsoln)

        sect = 'solver-time-integrator'
        f, i = cfg.getfloat, cfg.getint

        self._rtol_min = f(sect, 'krylov-rtol-min', 1e-3)
        self._rtol_max = f(sect, 'krylov-rtol-max', 1e-1)
        self.max_retries = i(sect, 'krylov-max-retries', 2)

        self._gp = LogGPOptimiser(10, (self._rtol_min, self._rtol_max))
        self._in_soft_reset = False

        sd = initsoln.state.get('intg/krylov-gp') if initsoln else None

        # Restart from saved state; skip warmup
        if sd is not None:
            self._best_tol = sd
            self._nwarmup = 0
            self._warmup_x = None
            self._probe_interval = self._base_interval
            self._stable_count = 2
        # Fresh start; warmup with geomspace sweep
        else:
            self._best_tol = self._rtol_max
            self._nwarmup = 5
            self._warmup_x = np.geomspace(self._rtol_max, self._rtol_min,
                                          self._nwarmup)
            self._probe_interval = 0
            self._stable_count = 0

        serialiser.register('intg/krylov-gp', lambda: self._best_tol or 0.0)

    @property
    def settled(self):
        return self._stable_count >= 2

    def _finalize_probe(self):
        if not self._probe_costs:
            return

        cost = np.median(self._probe_costs)
        self._gp.record(self._probe_tol, cost)
        self._probe_costs = []

        if self._gp.n < self._nwarmup:
            return

        new_best = self._gp.optimum(minimise=True)
        if new_best is None:
            new_best = self._best_tol

        # During soft reset update incumbent but freeze stability state
        if self._in_soft_reset:
            self._best_tol = new_best
            if self._gp.n >= self._nwarmup + 1:
                self._in_soft_reset = False

                # If the optimum shifted significantly, throughput data
                # is stale; drop settled so throughput re-explores
                if self._pre_soft_tol is not None:
                    shift = abs(np.log10(new_best / self._pre_soft_tol))
                    if shift > 0.3:
                        self._stable_count = 0
            return

        if self._best_tol is not None and new_best is not None:
            shift = abs(np.log10(new_best / self._best_tol))
            if shift < 0.3:
                self._probe_interval = np.clip(2*self._probe_interval,
                                               self._base_interval, 2000)
                self._stable_count += 1
            elif shift > 1.0:
                self._probe_interval = self._base_interval
                self._stable_count = 0
            else:
                self._stable_count = 0

        self._best_tol = new_best

    def _next_probe(self):
        # Warmup, probe at predetermined points covering the range
        if self._gp.n < self._nwarmup:
            wx = self._warmup_x

            # Early exit if optimum is near the loose end
            if ((opt := self._gp.optimum(minimise=True)) is not None and
                opt > wx[0]**0.75*wx[self._gp.n - 1]**0.25):
                self._probe_interval = self._base_interval
                return self._start_probe(self._best_tol)
            # Otherwise, probe at the next predetermined point
            else:
                return self._start_probe(wx[self._gp.n])
        # Post-warmup: GP-guided exploration
        else:
            explore = self._gp.optimum(minimise=True, explore=True)
            probe_tol = explore if explore is not None else self._best_tol

            self._steps_until_probe = self._probe_interval
            return self._start_probe(probe_tol)

    def reset(self):
        self._best_tol = self._rtol_max
        self._steps_until_probe = 0
        self._stable_count = 0
        self._gp.reset((self._rtol_min, self._rtol_max))
        self._probe_interval = 0
        self._in_soft_reset = False
        self._abort_probe()

    def soft_reset(self):
        self._pre_soft_tol = self._best_tol
        self._gp.reset()
        self._probe_interval = self._base_interval
        self._steps_until_probe = 0
        self._in_soft_reset = True
        self._abort_probe()


class ListController(BaseProbeController):
    name = 'list'

    def __init__(self, cfg, serialiser, initsoln):
        super().__init__(cfg, serialiser, initsoln)

        sect = 'solver-time-integrator'

        self._tols = cfg.getliteral(sect, 'krylov-rtol-list')
        self._reprobe_interval = cfg.getint(sect, 'krylov-reprobe-interval',
                                            200)

        self._rtol_max = max(self._tols)
        self._best_tol = self._rtol_max

        self._cycle_idx = 0
        self._tol_costs = {}
        self._settled = False

    @property
    def settled(self):
        return self._settled

    def _finalize_probe(self):
        if not self._probe_costs:
            return

        cost = np.median(self._probe_costs)
        self._tol_costs[self._probe_tol] = cost
        self._probe_costs = []

        # Once all tolerances have been probed, pick the best
        if len(self._tol_costs) == len(self._tols):
            self._best_tol = min(self._tol_costs, key=self._tol_costs.get)
            self._settled = True

    def _next_probe(self):
        # Still cycling through the list
        if self._cycle_idx < len(self._tols):
            tol = self._tols[self._cycle_idx]
            self._cycle_idx += 1
            return self._start_probe(tol)
        # All probed; schedule re-probe and return incumbent
        else:
            self._cycle_idx = 0
            self._tol_costs = {}
            self._steps_until_probe = self._reprobe_interval
            return self._best_tol

    def reset(self):
        self._best_tol = self._rtol_max
        self._steps_until_probe = 0
        self._cycle_idx = 0
        self._tol_costs = {}
        self._settled = False
        self._abort_probe()

    def soft_reset(self):
        self._steps_until_probe = 0
        self._cycle_idx = 0
        self._tol_costs = {}
        self._abort_probe()
