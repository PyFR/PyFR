# -*- coding: utf-8 -*-

import math

from pyfr.integrators.dual.phys.base import BaseDualIntegrator


class BaseDualStepper(BaseDualIntegrator):
    pass


class BaseDIRKStepper(BaseDualStepper):
    stepper_nregs = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.fsal:
            self.b = self.a[-1]

        self.c = [sum(row) for row in self.a]

    @property
    def stage_nregs(self):
        return self.nstages

    def step(self, t, dt):
        for s, (sc, tc) in enumerate(zip(self.a, self.c)):
            self.pseudointegrator.init_stage(s, sc, dt)
            self.pseudointegrator.pseudo_advance(t + dt*tc)

        self._finalize_step()

    def _finalize_step(self):
        if not self.fsal:
            bcoeffs = [bt*self._dt for bt in self.b]
            self.pseudointegrator.obtain_solution(bcoeffs)

        self.pseudointegrator.store_current_soln()


class DualBackwardEulerStepper(BaseDIRKStepper):
    stepper_name = 'backward-euler'
    nstages = 1
    fsal = True

    a = [[1]]


class SDIRK33Stepper(BaseDIRKStepper):
    stepper_name = 'sdirk33'
    nstages = 3
    fsal = True

    _at = math.atan(0.5**1.5)/3
    _a_lam = (3**0.5*math.sin(_at) - math.cos(_at))/2**0.5 + 1

    a = [
        [_a_lam],
        [0.5*(1 - _a_lam), _a_lam],
        [(4 - 1.5*_a_lam)*_a_lam - 0.25, (1.5*_a_lam - 5)*_a_lam + 1.25, _a_lam]
    ]


class SDIRK43Stepper(BaseDIRKStepper):
    stepper_name = 'sdirk43'
    nstages = 3
    fsal = False

    _a_lam = (3 + 2*3**0.5*math.cos(math.pi/18))/6

    a = [
        [_a_lam],
        [0.5 - _a_lam,  _a_lam],
        [2*_a_lam, 1 - 4*_a_lam, _a_lam]
    ]

    _b_rlam = 1/(6*(1 - 2*_a_lam)*(1 - 2*_a_lam))
    b = [_b_rlam, 1 - 2*_b_rlam, _b_rlam]
