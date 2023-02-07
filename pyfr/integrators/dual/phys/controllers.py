import numpy as np

from pyfr.integrators.dual.phys.base import BaseDualIntegrator


class BaseDualController(BaseDualIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            for csh in self.completed_step_handlers:
                csh(self)

        np.random.seed(0)   

    def _accept_step(self, dt, idxcurr):
        self.tcurr += dt
        self.nacptsteps += 1
        self.nacptchain += 1

        # Filter
        if self._fnsteps and self.nacptsteps % self._fnsteps == 0:
            self.pseudointegrator.system.filt(idxcurr)

        # Invalidate the solution cache
        self._curr_soln = None

        # Invalidate the solution gradients cache
        self._curr_grad_soln = None

        # Fire off any event handlers
        for csh in self.completed_step_handlers:
            csh(self)

        # Abort if plugins request it
        self._check_abort()

        # Clear the pseudo step info
        self.pseudointegrator.pseudostepinfo = []


class DualNoneController(BaseDualController):
    controller_name = 'none'

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:

            if self.tcurr + 2*self._dt < t < self.tcurr + 3*self._dt:
                dt = 0.5*(t - self.tcurr)
            elif self.tcurr + self._dt < t < self.tcurr + 2*self._dt:
                dt = t - self.tcurr
            else:
                dt = self._dt

            if self.pseudointegrator.dt != dt:
                # Change dt in pseudo-integrator (and multi-p levels)
                
                self.pseudointegrator.dt = dt

            # Take the physical step
            self.step(self.tcurr, dt)

            # We are not adaptive, so accept every step
            self._accept_step(dt, self.pseudointegrator._idxcurr)
