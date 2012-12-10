# -*- coding: utf-8 -*-

from pyfr.integrators.base import BaseIntegrator


class BaseController(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super(BaseController, self).__init__(*args, **kwargs)

        # Bank index of solution
        self._idxcurr = 0


class NoneController(BaseController):
    controller_name = 'none'

    def __init__(self, *args, **kwargs):
        super(NoneController, self).__init__(*args, **kwargs)

        self._dt = self._cfg.getfloat('time-integration', 'dt')
        self._dtmin = 0.1*self._dt

    @property
    def _controller_needs_errest(self):
        return False

    def advance_to(self, t):
        if t < self._tcurr:
            raise ValueError('Advance time is in the past')

        while (t - self._tcurr) > self._dtmin:
            # Decide on the time step
            dt = min(t - self._tcurr, self._dt)

            # Take the step
            self._idxcurr = self.step(self._tcurr, dt)

            # Increment the time
            self._tcurr += dt

        # Return the solution matrices
        return self._meshp.ele_scal_upts(self._idxcurr)
