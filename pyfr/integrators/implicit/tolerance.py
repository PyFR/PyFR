import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.nputil import LogGPOptimiser
from pyfr.util import subclass_where


def get_linear_tol_controller(cfg, serialiser, initsoln, nimplicit,
                              nl_maxiter):
    sect = 'solver-time-integrator'
    name = cfg.get(sect, 'linear-tol-controller', 'none')
    cls = subclass_where(BaseToleranceController, name=name)

    return cls(cfg, serialiser, initsoln, nimplicit, nl_maxiter)


class BaseToleranceController:
    name = None
    max_retries = 0
    probing = False
    probing_other = False
    _best_tol = None

    # Log cost of the incumbent relative to the first incumbent
    relative_cost = 0.0

    # Whether Newton may loosen the tolerance to avoid oversolving
    safeguard = False

    @property
    def best_tol(self):
        return self._best_tol

    def update(self, wall_time, gamma_dt, success, niters):
        pass

    def begin_step(self):
        pass

    def fit_window(self, nsolves):
        pass


class NullToleranceController(BaseToleranceController):
    name = 'none'

    def __init__(self, cfg, serialiser, initsoln, nimplicit,
                 nl_maxiter):
        sect = 'solver-time-integrator'
        self._best_tol = self._tol = cfg.getfloat(sect, 'linear-rtol', 1e-2)

    def select_tolerance(self, gamma_dt, build):
        return self._tol


class BaseProbeController(BaseToleranceController):
    safeguard = True

    # Decades of gamma*dt drift after which the tolerances are re-probed
    _gdt_shift = 0.5

    def __init__(self, cfg, serialiser, initsoln, nimplicit,
                 nl_maxiter):
        sect = 'solver-time-integrator'
        self._probe_nsolves = cfg.getint(sect, 'linear-probe-nsolves', 20)
        self._best_tol = cfg.getfloat(sect, 'linear-rtol', 1e-2)

        # Implicit solves per time step
        self._nimplicit = nimplicit

        # Nonlinear iteration cap of the solver
        self._nl_maxiter = nl_maxiter

        # Set once a probe has seen solves needing more than one iteration
        self._stiff = False

        # gamma*dt the cost model belongs to
        self._model_gdt = None

        # Operator and preconditioner build the probe in flight belongs to
        self._ctl_key = None

        # Tolerance handed to the most recent solve
        self._tol = self._best_tol

        self._steps_until_probe = 0
        self._solves_in_step = 0

        # Start with no probe in flight
        self._drop_probes()

        # Seeded so that every rank draws the same probe order
        self._rng = np.random.default_rng(0)

    def fit_window(self, nsolves):
        # Limit a probe, incumbent steps included, to half a window of solves
        self._probe_nsolves = max(nsolves // 2, 1)

    @property
    def probing(self):
        return self._probe_tol is not None

    @property
    def probing_other(self):
        return self.probing and not self._is_incumbent(self._tol)

    def _is_incumbent(self, tol):
        # Treat tolerances within 5% of the incumbent as the incumbent
        return abs(np.log(tol / self._best_tol)) < 0.05

    def _drop_probes(self):
        # Discard any probe in flight along with its incumbent steps
        self._probe_tol = None
        self._probe_order = []
        self._period_costs = []
        self._period_means = []
        self._probe_iters = []

    def _start_probe(self, tol):
        self._drop_probes()

        # Size the probe in step pairs and draw the order within each pair
        npairs = max(self._probe_nsolves // (2*self._nimplicit), 1)
        self._probe_tol = tol
        self._probe_order = list(self._rng.integers(2, size=npairs))
        return self._period_tol()

    def _period_tol(self):
        # Tolerance of the current step, as drawn for its pair
        i = len(self._period_means)
        if self._probe_order[i // 2] == i % 2:
            return self._probe_tol
        else:
            return self._best_tol

    def begin_step(self):
        self._solves_in_step = 0

        # Close the previous step's period of the probe
        if self.probing and self._period_costs:
            self._close_period()

    def _close_period(self):
        self._period_means.append(np.mean(self._period_costs))
        self._period_costs = []

        # Every pair has both steps costed; evaluate the probe
        if len(self._period_means) == 2*len(self._probe_order):
            self._end_probe()

    def _adopt(self, tol, rel_cost):
        # Switch incumbent and add its cost to the running relative cost
        self._best_tol = tol
        self.relative_cost += rel_cost

    def _end_probe(self):
        tol, order = self._probe_tol, self._probe_order
        means, iters = self._period_means, self._probe_iters
        self._drop_probes()

        # Discard a probe of trivial solves and wait before probing again
        trivial = np.mean(iters) <= 1 and self._nl_maxiter > 1
        if not self._stiff and trivial:
            self._steps_until_probe = 2*self._probe_nsolves
            return
        else:
            self._stiff = True

        # Mean incumbent cost stands in for a partner step a failure cut short
        inc = [m for i, m in enumerate(means) if order[i // 2] != i % 2]
        ref = np.mean(inc) if inc else None

        # Difference of each pair: candidate step cost less incumbent step cost
        diffs = []
        for k, o in enumerate(order[:(len(means) + 1) // 2]):
            p, c = 2*k + o, 2*k + 1 - o
            if c < len(means):
                diffs.append(means[p] - means[c])
            elif p < len(means) and ref is not None:
                diffs.append(means[p] - ref)
            elif p < len(means):
                diffs.append(1.0)

        if not diffs:
            return

        # Probing the incumbent against itself only measures the noise
        cost = 0.0 if self._is_incumbent(tol) else np.mean(diffs)
        self._finalize_probe(tol, cost, self._mean_var(diffs))

    def _mean_var(self, costs):
        # Sampling variance of the mean of a block of costs
        if len(costs) > 1:
            return np.var(costs, ddof=1) / len(costs)
        else:
            return float('nan')

    def update(self, wall_time, gamma_dt, success, niters):
        # Log cost per unit gamma*dt, plus a decade of penalty on failure
        cost = np.log10(max(wall_time / gamma_dt, 1e-10)) + (not success)

        # Accumulate cost during probe window
        if self.probing:
            self._period_costs.append(cost)
            self._probe_iters.append(niters)

            # Finish early on failure so the penalised cost is recorded
            if not success:
                self._close_period()
                if self.probing:
                    self._end_probe()

    def select_tolerance(self, gamma_dt, build):
        # Refresh the model promptly once the operator has moved away
        if self._model_gdt is None:
            self._model_gdt = gamma_dt
        elif abs(np.log10(gamma_dt / self._model_gdt)) > self._gdt_shift:
            self._model_gdt = gamma_dt
            self._refresh()

        # Both halves of a pair must share an operator and preconditioner
        if (gamma_dt, build) != self._ctl_key:
            self._ctl_key = (gamma_dt, build)

            # Retry a probe cut short after a delay out of step with builds
            if self.probing:
                delay = self._rng.integers(2*self._probe_nsolves)
                self._steps_until_probe = int(delay)

            self._drop_probes()

        # Currently in a probe; use the tolerance assigned to this step
        if self.probing:
            self._tol = self._period_tol()
        # Waiting between probes; return incumbent
        elif self._steps_until_probe > 0:
            self._steps_until_probe -= 1
            self._tol = self._best_tol
        # Later solves of a step take the incumbent; a probe opens on the first
        elif self._solves_in_step > 0:
            self._tol = self._best_tol
        else:
            self._tol = self._next_probe()

        self._solves_in_step += 1

        return self._tol


class WindowedGPController(BaseProbeController):
    name = 'windowed-gp'
    max_retries = 2
    _base_interval = 100

    def __init__(self, cfg, serialiser, initsoln, nimplicit,
                 nl_maxiter):
        super().__init__(cfg, serialiser, initsoln, nimplicit,
                         nl_maxiter)

        sect = 'solver-time-integrator'
        f, i = cfg.getfloat, cfg.getint

        self._rtol_min = f(sect, 'linear-rtol-min', 1e-3)
        self._rtol_max = f(sect, 'linear-rtol-max', 1e-1)
        self.max_retries = i(sect, 'linear-max-retries', 2)

        # Cost model whose observations age out as the flow develops
        self._gp = LogGPOptimiser(10, (self._rtol_min, self._rtol_max),
                                  decay=10)

        sd = initsoln.state.get('intg/krylov-gp') if initsoln else None

        # Restart from saved state; short warmup around the saved tolerance
        if sd is not None:
            self._best_tol = sd
            self.reset([sd*10**0.5, sd*10**-0.5])
        # Fresh start; warmup with geomspace sweep
        else:
            self.reset()

        serialiser.register('intg/krylov-gp', lambda: self._best_tol or 0.0)

    def _finalize_probe(self, tol, cost, noise_var):
        # Enter the probe into the model and adopt its optimum once warmed up
        self._gp.update(tol, cost, noise_var)

        if self._gp.n >= self._nwarmup:
            self._adopt_optimum()

    def _adopt_optimum(self):
        comm, rank, root = get_comm_rank_root()

        new_best = self._gp.optimum(minimise=True)
        if new_best is None:
            return False

        # Switch only when the predicted gain exceeds its uncertainty
        rel_cost, var_new = self._gp.predict(new_best)
        inc_cost, var_inc = self._gp.predict(self._best_tol)
        adopt = inc_cost - rel_cost > (var_new + var_inc)**0.5

        # Take the root's decision so that every rank keeps the same model
        new_best, rel_cost, adopt = comm.bcast((new_best, rel_cost, adopt),
                                               root=root)
        if adopt:
            self._gp.shift(-rel_cost)
            self._adopt(new_best, rel_cost)
            return True
        else:
            return False

    def _next_probe(self):
        comm, rank, root = get_comm_rank_root()

        # Warmup, probe at predetermined points covering the range
        if self._gp.n < self._nwarmup:
            probed = self._warmup_x[:self._gp.n]

            # Early exit if optimum is near the loose end
            opt = self._gp.optimum(minimise=True)
            knee = max(probed)**0.75*min(probed)**0.25
            loose = opt is not None and opt > knee

            # Take the root's verdict so that every rank probes the same point
            loose = comm.bcast(loose, root=root)

            # Provided the loose end is a significant enough gain to adopt
            if loose and self._adopt_optimum():
                self._nwarmup = self._gp.n
                self._steps_until_probe = self._base_interval
                return self._best_tol
            # Otherwise, probe at the next predetermined point
            else:
                return self._start_probe(self._warmup_x[self._gp.n])
        # Post-warmup: refresh whichever observation has aged the most
        else:
            self._steps_until_probe = self._base_interval
            return self._start_probe(self._gp.stalest())

    def reset(self, sweep=None):
        # Sweep the range afresh, referenced to the incumbent at zero cost
        lo, hi = self._rtol_min, self._rtol_max
        if sweep is None:
            sweep = np.geomspace(hi, lo, 5)

        self._best_tol = tol0 = np.clip(self._best_tol, lo, hi)
        sweep = np.clip(sweep, lo, hi)
        far = [x for x in sweep if not self._is_incumbent(x)]
        self._warmup_x = [tol0, *far]
        self._nwarmup = len(self._warmup_x)

        self._steps_until_probe = 0
        self._gp.reset((lo, hi))
        self._gp.update(tol0, 0.0)
        self._drop_probes()

    def _refresh(self):
        # Keep the model but bring the next probe forward
        self._steps_until_probe = 0


class ListController(BaseProbeController):
    name = 'list'

    def __init__(self, cfg, serialiser, initsoln, nimplicit,
                 nl_maxiter):
        super().__init__(cfg, serialiser, initsoln, nimplicit,
                         nl_maxiter)

        sect = 'solver-time-integrator'

        self._tols = cfg.getliteral(sect, 'linear-rtol-list')
        self._reprobe_interval = cfg.getint(sect, 'linear-reprobe-interval',
                                            200)

        self.reset()

    def _finalize_probe(self, tol, cost, noise_var):
        self._tol_costs[tol] = cost

        # Once all tolerances have been probed, pick the best
        if len(self._tol_costs) == len(self._tols):
            new_best = min(self._tol_costs, key=self._tol_costs.get)
            self._adopt(new_best, self._tol_costs[new_best])

    def _next_probe(self):
        # Still cycling through the list
        if self._cycle_idx < len(self._tols):
            tol = self._tols[self._cycle_idx]
            self._cycle_idx += 1
            return self._start_probe(tol)
        # All probed; schedule re-probe and return incumbent
        else:
            self._refresh()
            self._steps_until_probe = self._reprobe_interval
            return self._best_tol

    def reset(self):
        # Start from the listed tolerance nearest the incumbent
        tol0 = self._best_tol
        self._best_tol = min(self._tols, key=lambda t: abs(np.log(t / tol0)))
        self._refresh()

    def _refresh(self):
        # Probe the incumbent first to give each cycle its reference cost
        self._tols.sort(key=lambda t: t != self._best_tol)
        self._steps_until_probe = 0
        self._cycle_idx = 0
        self._tol_costs = {}
        self._drop_probes()
