# -*- coding: utf-8 -*-

from abc import abstractmethod

from pyfr.integrators.base import BaseIntegrator


class BaseDualIntegrator(BaseIntegrator):
    formulation = 'dual'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dtau = self.cfg.getfloat('solver-time-integrator', 'pseudo-dt')
        self.dtaumin = 1.0e-12

    @property
    def _stepper_regidx(self):
        return self._regidx[:self._pseudo_stepper_nregs]

    @property
    def _source_regidx(self):
        return self._regidx[self._pseudo_stepper_nregs:]

    @abstractmethod
    def _dual_time_source(self):
        pass

    @abstractmethod
    def finalise_step(self, currsoln):
        pass
