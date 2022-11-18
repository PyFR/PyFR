import math

from pyfr.integrators.base import BaseIntegrator
from pyfr.integrators.dual.pseudo import get_pseudo_integrator


class BaseDualIntegrator(BaseIntegrator):
    formulation = 'dual'

    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        super().__init__(backend, rallocs, mesh, initsoln, cfg)

        self._saved_soln = None
        self.rewind = None

        # Get the pseudo-integrator
        self.pseudointegrator = get_pseudo_integrator(
            backend, systemcls, rallocs, mesh, initsoln, cfg,
            self.stepper_nregs, self.stage_nregs, self._dt
        )

        # Event handlers for advance_to
        self.completed_step_handlers = self._get_plugins(initsoln)

        # Delete the memory-intensive elements map from the system
        del self.system.ele_map

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

    def save_soln(self):
        if self.rewind is False:
            self._saved_soln = self.soln
        else:
            raise Exception('Rewind is not set to False.')

    @property
    def saved_soln(self):
        return self._saved_soln

    @saved_soln.setter
    def saved_soln(self, y):
        self._saved_soln = y

    def rewind_soln(self):
        if self.saved_soln and self.rewind is True:
            self.system.ele_scal_upts_set(self.pseudointegrator._stepper_regidx, self.saved_soln)
        else:
            raise Exception('No saved solution to load, or rewind is not set to True.')

    @property
    def grad_soln(self):
        system = self.system

        if not self._curr_grad_soln:
            system.compute_grads(self.tcurr, self.pseudointegrator._idxcurr)
            self._curr_grad_soln = [e.get() for e in system.eles_vect_upts]

        return self._curr_grad_soln

    def call_plugin_dt(self, dt):
        rem = math.fmod(dt, self._dt)
        tol = 5.0*self.dtmin
        if rem > tol and (self._dt - rem) > tol:
            raise ValueError('Plugin call times must be multiples of dt')

        super().call_plugin_dt(dt)

    def collect_stats(self, stats):
        super().collect_stats(stats)

        self.pseudointegrator.collect_stats(stats)
