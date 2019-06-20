# -*- coding: utf-8 -*-

from pyfr.integrators.dual.phys.base import BaseDualIntegrator


class BaseDualStepper(BaseDualIntegrator):
    pass


class DualBDF2Stepper(BaseDualStepper):
    stepper_name = 'bdf2'

    @property
    def _stepper_order(self):
        return 2

    @property
    def _stepper_coeffs(self):
        return [-1.5, 2.0, -0.5]


class DualBDF3Stepper(BaseDualStepper):
    stepper_name = 'bdf3'

    @property
    def _stepper_order(self):
        return 3

    @property
    def _stepper_coeffs(self):
        return [-11.0/6.0, 3.0, -1.5, 1.0/3.0]


class DualBackwardEulerStepper(BaseDualStepper):
    stepper_name = 'backward-euler'

    @property
    def _stepper_order(self):
        return 1

    @property
    def _stepper_coeffs(self):
        return [-1.0, 1.0]
