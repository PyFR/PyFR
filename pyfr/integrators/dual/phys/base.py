import math

from pyfr.integrators.base import BaseIntegrator
from pyfr.integrators.dual.pseudo import get_pseudo_integrator


class BaseDualIntegrator(BaseIntegrator):
    formulation = 'dual'

    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        super().__init__(backend, rallocs, mesh, initsoln, cfg)

        # Get the pseudo-integrator
        self.pseudointegrator = get_pseudo_integrator(
            backend, systemcls, rallocs, mesh, initsoln, cfg,
            self.stepper_nregs, self.stage_nregs, self.dt
        )

        # Event handlers for advance_to
        self.plugins = self._get_plugins(initsoln)

        # Commit the pseudo integrators now we have the plugins
        self.pseudointegrator.commit()

    @property
    def system(self):
        return self.pseudointegrator.system

    @property
    def pseudostepinfo(self):
        return self.pseudointegrator.pseudostepinfo

    @property
    def soln(self):
        if not self._curr_soln:
            self._curr_soln = self.system.ele_scal_upts(
                self.pseudointegrator._idxcurr
            )

        return self._curr_soln

    @property
    def grad_soln(self):
        system = self.system

        if not self._curr_grad_soln:
            system.compute_grads(self.tcurr, self.pseudointegrator._idxcurr)
            self._curr_grad_soln = [e.get() for e in system.eles_vect_upts]

        return self._curr_grad_soln

    @property
    def dt_soln(self):
        system = self.system

        if not self._curr_dt_soln:
            copy = self.soln
            idx = self.pseudointegrator._idxcurr
            system.rhs(self.tcurr, idx, idx)
            self._curr_dt_soln = system.ele_scal_upts(idx)

            # Reset current register with original contents
            for c, e in zip(copy, system.ele_banks):
                e[idx].set(c)

        return self._curr_dt_soln

    def call_plugin_dt(self, dt):
        rem = math.fmod(dt, self.dt)
        tol = 5.0*self.dtmin
        if rem > tol and (self.dt - rem) > tol:
            raise ValueError('Plugin call times must be multiples of dt')

        super().call_plugin_dt(dt)

    def collect_stats(self, stats):
        super().collect_stats(stats)

        self.pseudointegrator.collect_stats(stats)
