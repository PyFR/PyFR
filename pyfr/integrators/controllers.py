# -*- coding: utf-8 -*-

from pyfr.integrators.base import BaseIntegrator
from pyfr.util import proxylist


class BaseController(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super(BaseController, self).__init__(*args, **kwargs)

        # Current and minimum time steps
        self._dt = self._cfg.getfloat('solver-time-integrator', 'dt')
        self._dtmin = 1.0e-14

        # Bank index of solution
        self._idxcurr = 0

        # Accepted and rejected step counters
        self.nacptsteps = 0
        self.nrjctsteps = 0
        self.nacptchain = 0

        # Event handlers for advance_to
        self.completed_step_handlers = proxylist([])

    @property
    def nsteps(self):
        return self.nacptsteps + self.nrjctsteps

    @property
    def soln(self):
        return self._system.ele_scal_upts(self._idxcurr)


class NoneController(BaseController):
    controller_name = 'none'

    def __init__(self, *args, **kwargs):
        super(NoneController, self).__init__(*args, **kwargs)

    @property
    def _controller_needs_errest(self):
        return False

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            # Decide on the time step
            dt = max(min(t - self.tcurr, self._dt), self._dtmin)

            # Take the step
            self._idxcurr = self.step(self.tcurr, dt)

            # Increment the time
            self.tcurr += dt

            # Update status
            self.nacptsteps += 1
            self.nacptchain += 1

            # Fire off any event handlers
            self.completed_step_handlers(self)

        # Return the solution matrices
        return self.soln
