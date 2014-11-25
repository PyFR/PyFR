# -*- coding: utf-8 -*-

import math

from pyfr.integrators.base import BaseIntegrator
from pyfr.util import memoize, proxylist


class BaseController(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Current and minimum time steps
        self._dt = self.cfg.getfloat('solver-time-integrator', 'dt')
        self._dtmin = 1.0e-14

        # Solution filtering frequency
        self._ffreq = self.cfg.getint('soln-filter', 'freq', '0')

        # Bank index of solution
        self._idxcurr = 0

        # Accepted and rejected step counters
        self.nacptsteps = 0
        self.nrjctsteps = 0
        self.nacptchain = 0

        # Event handlers for advance_to
        self.completed_step_handlers = proxylist([])

    def _accept_step(self, dt, idxcurr):
        self.tcurr += dt
        self.nacptsteps += 1
        self.nacptchain += 1

        self._idxcurr = idxcurr

        # Filter
        if self._ffreq and self.nacptsteps % self._ffreq == 0:
            self._system.filt(idxcurr)

        # Fire off any event handlers
        self.completed_step_handlers(self)

    def _reject_step(self, dt, idxold):
        if dt <= self._dtmin:
            raise RuntimeError('Minimum sized time step rejected')

        self.nacptchain = 0
        self.nrjctsteps += 1

        self._idxcurr = idxold

    @property
    def nsteps(self):
        return self.nacptsteps + self.nrjctsteps

    @property
    def soln(self):
        return self._system.ele_scal_upts(self._idxcurr)


class NoneController(BaseController):
    controller_name = 'none'

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
            idxcurr = self.step(self.tcurr, dt)

            # We are not adaptive, so accept every step
            self._accept_step(dt, idxcurr)

        # Return the solution matrices
        return self.soln


class PIController(BaseController):
    controller_name = 'pi'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = 'solver-time-integrator'

        # Error tolerances
        self._atol = self.cfg.getfloat(sect, 'atol')
        self._rtol = self.cfg.getfloat(sect, 'rtol')

        # PI control values
        self._alpha = self.cfg.getfloat(sect, 'pi-alpha', 0.7)
        self._beta = self.cfg.getfloat(sect, 'pi-beta', 0.4)

        # Estimate of previous error
        self._errprev = 0.0

        # Step size adjustment factors
        self._saffac = self.cfg.getfloat(sect, 'safety-fact', 0.8)
        self._maxfac = self.cfg.getfloat(sect, 'max-fact', 2.5)
        self._minfac = self.cfg.getfloat(sect, 'min-fact', 0.3)

    @property
    def _controller_needs_errest(self):
        return True

    @memoize
    def _get_errest_kerns(self):
        return self._kernel('errest', nargs=3)

    def _errest(self, x, y, z):
        from mpi4py import MPI

        errest = self._get_errest_kerns()

        # Obtain an estimate for the error
        self._prepare_reg_banks(x, y, z)
        self._queue % errest(self._atol, self._rtol)

        # Reduce locally (element types) and globally (MPI ranks)
        rl = sum(errest.retval)
        rg = MPI.COMM_WORLD.allreduce(rl, op=MPI.SUM)

        # Normalise
        err = math.sqrt(rg / self._gndofs)

        return err if not math.isnan(err) else 100

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            # Decide on the time step
            dt = max(min(t - self.tcurr, self._dt), self._dtmin)

            # Take the step
            idxcurr, idxprev, idxerr = self.step(self.tcurr, dt)

            # Estimate the error
            err = self._errest(idxerr, idxcurr, idxprev)

            maxf = self._maxfac
            minf = self._minfac
            saff = self._saffac
            sord = self._stepper_order

            expa = self._alpha / self._stepper_order
            expb = self._beta / self._stepper_order

            # Determine time step adjustment factor
            fac = err**-expa*self._errprev**expb
            fac = min(maxf, max(minf, saff*fac))

            # Compute the size of the next step
            self._dt = fac*dt

            # Decide if to accept or reject the step
            if err < 1.0:
                self._errprev = err
                self._accept_step(dt, idxcurr)
            else:
                self._reject_step(dt, idxprev)

        return self.soln
