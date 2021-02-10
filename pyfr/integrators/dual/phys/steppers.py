# -*- coding: utf-8 -*-

from pyfr.integrators.dual.phys.base import BaseDualIntegrator


class BaseDualStepper(BaseDualIntegrator):
    pass


class DualBDF2Stepper(BaseDualStepper):
    stepper_name = 'bdf2'
    stepper_order = 2
    stepper_coeffs = [-1.5, 2.0, -0.5]


class DualBDF3Stepper(BaseDualStepper):
    stepper_name = 'bdf3'
    stepper_order = 3
    stepper_coeffs = [-11.0/6.0, 3.0, -1.5, 1.0/3.0]


class DualBackwardEulerStepper(BaseDualStepper):
    stepper_name = 'backward-euler'
    stepper_order = 1
    stepper_coeffs = [-1.0, 1.0]
