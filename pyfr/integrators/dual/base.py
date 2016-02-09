# -*- coding: utf-8 -*-

from pyfr.integrators.base import BaseIntegrator


class BaseDualIntegrator(BaseIntegrator):
    formulation = 'dual'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pass
