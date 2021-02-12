# -*- coding: utf-8 -*-

import math

from pyfr.integrators.base import BaseIntegrator
from pyfr.integrators.dual.pseudo import get_pseudo_integrator
from pyfr.util import proxylist


class BaseDualIntegrator(BaseIntegrator):
    formulation = 'dual'

    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        super().__init__(backend, rallocs, mesh, initsoln, cfg)

        # Get the pseudo-integrator
        self.pseudointegrator = get_pseudo_integrator(
            backend, systemcls, rallocs, mesh,
            initsoln, cfg, self.stepper_coeffs, self._dt
        )

        # Event handlers for advance_to
        self.completed_step_handlers = proxylist(self._get_plugins())

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
        # If we do not have the solution cached then fetch it
        if not self._curr_soln:
            self._curr_soln = self.system.ele_scal_upts(
                self.pseudointegrator._idxcurr
            )

        return self._curr_soln

    @property
    def grad_soln(self):
        # If we do not have the solution gradients cached then compute and fetch them
        if not self._curr_grad_soln:
            self.system.compute_grads(self.tcurr, self.pseudointegrator._idxcurr)
            self._curr_grad_soln = self.system.eles_vect_upts.get()

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
