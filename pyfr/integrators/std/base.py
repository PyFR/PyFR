# -*- coding: utf-8 -*-

from abc import abstractproperty

from pyfr.integrators.base import BaseIntegrator
from pyfr.util import proxylist

class BaseStdIntegrator(BaseIntegrator):
    formulation = 'std'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nreg = self._stepper_nregs

        # Storage register banks
        self._regs, self._regidx = self._get_reg_banks(nreg)

        # Bank index of solution
        self._idxcurr = 0

        # Sanity checks
        if self._controller_needs_errest and not self._stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Event handlers for advance_to
        self.completed_step_handlers = proxylist(self._get_plugins())

        # Delete the memory-intensive elements map from the system
        del self.system.ele_map

    @abstractproperty
    def _controller_needs_errest(self):
        pass

    @abstractproperty
    def _stepper_has_errest(self):
        pass
