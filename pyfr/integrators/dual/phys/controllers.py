from pyfr.integrators.dual.phys.base import BaseDualIntegrator


class BaseDualController(BaseDualIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        self.i = self.cfg.getint('solver-time-integrator', 'dt-switch', 2)

        if self.i<=1:
            raise ValueError('dt-switch must be greater than 1')

        self._if_near = False
        self._dt_near = self._dt

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()

    def _accept_step(self, idxcurr):
        self.tcurr += self.dt
        self.nacptsteps += 1
        self.nacptchain += 1

        # Filter
        if self._fnsteps and self.nacptsteps % self._fnsteps == 0:
            self.pseudointegrator.system.filt(idxcurr)

        # Invalidate the solution cache
        self._curr_soln = None

        # Invalidate the solution gradients cache
        self._curr_grad_soln = None

        # Run any plugins
        self._run_plugins()

        # Clear the pseudo step info
        self.pseudointegrator.pseudostepinfo = []


class DualNoneController(BaseDualController):
    controller_name = 'none'
    controller_has_variable_dt = False

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            if self.tcurr + self._dt <= t < self.tcurr + self.i*self._dt:
                if not self._if_near:
                    self._dt_near = (t-self.tcurr)/((t-self.tcurr)//self._dt + 1.0)
                    self._if_near = True
                self.dt = self._dt_near
            elif self.tcurr < t <= self.tcurr + self._dt:
                self.dt = t - self.tcurr
            else:
                self.dt = self._dt
                self._if_near = False

            # Take the physical step
            self.step(self.tcurr, self.dt)

            # We are not adaptive, so accept every step
            self._accept_step(self.pseudointegrator._idxcurr)
