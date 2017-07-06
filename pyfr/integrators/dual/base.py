# -*- coding: utf-8 -*-

from pyfr.integrators.base import BaseIntegrator


class BaseDualIntegrator(BaseIntegrator):
    formulation = 'dual'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = 'solver-time-integrator'

        self._dtaumin = 1.0e-12
        self._dtau = self.cfg.getfloat(sect, 'pseudo-dt')

        self._maxniters = self.cfg.getint(sect, 'pseudo-niters-max', 0)
        self._minniters = self.cfg.getint(sect, 'pseudo-niters-min', 0)

        if self._maxniters < self._minniters:
            raise ValueError('The maximum number of pseudo-iterations must '
                             'be greater than or equal to the minimum')

        self._pseudo_residtol= self.cfg.getfloat(sect, 'pseudo-resid-tol')
        self._pseudo_norm = self.cfg.get(sect, 'pseudo-resid-norm', 'l2')

        if self._pseudo_norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid pseudo-residual norm')

    @property
    def _stepper_regidx(self):
        return self._regidx[:self._pseudo_stepper_nregs]

    @property
    def _source_regidx(self):
        pnreg, dsrc = self._pseudo_stepper_nregs, self._dual_time_source
        return self._regidx[pnreg:pnreg + len(dsrc) - 1]

    def _dual_time_source(self):
        pass

    def finalise_step(self, currsoln):
        pass
