# -*- coding: utf-8 -*-

from pyfr.integrators.base import BaseIntegrator


class BaseStdIntegrator(BaseIntegrator):
    formulation = 'std'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Sanity checks
        if self._controller_needs_errest and not self._stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

    @property
    def _controller_needs_errest(self):
        pass

    @property
    def _stepper_has_errest(self):
        pass
