import math

from pyfr.integrators.base import BaseIntegrator, _common_plugin_prop
from pyfr.integrators.dual.pseudo import get_pseudo_integrator


class BaseDualIntegrator(BaseIntegrator):
    formulation = 'dual'

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, mesh, initsoln, cfg)

        # Get the pseudo-integrator
        self.pseudointegrator = get_pseudo_integrator(
            backend, systemcls, mesh, initsoln, cfg, self.stepper_nregs,
            self.stage_nregs, self._dt
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

    @_common_plugin_prop('_curr_soln')
    def soln(self):
        return self.system.ele_scal_upts(self.pseudointegrator._idxcurr)

    @_common_plugin_prop('_curr_grad_soln')
    def grad_soln(self):
        self.system.compute_grads(self.tcurr, self.pseudointegrator._idxcurr)
        return [e.get() for e in self.system.eles_vect_upts]

    @_common_plugin_prop('_curr_dt_soln')
    def dt_soln(self):
        soln = self.soln

        idx = self.pseudointegrator._idxcurr
        self.system.rhs(self.tcurr, idx, idx)

        dt_soln = self.system.ele_scal_upts(idx)

        # Reset current register with original contents
        for e, s in zip(self.system.ele_banks, soln):
            e[idx].set(s)

        return dt_soln

    def call_plugin_dt(self, dt):
        rem = math.fmod(dt, self._dt)
        tol = 5.0*self.dtmin
        if rem > tol and (self._dt - rem) > tol:
            raise ValueError('Plugin call times must be multiples of dt')

        super().call_plugin_dt(dt)

    def collect_stats(self, stats):
        super().collect_stats(stats)

        self.pseudointegrator.collect_stats(stats)
