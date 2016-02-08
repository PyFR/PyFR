# -*- coding: utf-8 -*-

import math
import re
import time

from pyfr.integrators.base import BaseIntegrator
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins import get_plugin
from pyfr.util import memoize, proxylist


class BaseController(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Current and minimum time steps
        self._dt = self.cfg.getfloat('solver-time-integrator', 'dt')
        self.dtmin = 1.0e-12

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Bank index of solution
        self._idxcurr = 0

        # Solution cache
        self._curr_soln = None

        # Accepted and rejected step counters
        self.nacptsteps = 0
        self.nrjctsteps = 0
        self.nacptchain = 0

        # Stats on the most recent step
        self.stepinfo = []

        # Event handlers for advance_to
        self.completed_step_handlers = proxylist([])

        # Record the starting wall clock time
        self._wstart = time.time()

        # Load any plugins specified in the config file
        for s in self.cfg.sections():
            m = re.match('soln-plugin-(.+?)(?:-(.+))?$', s)
            if m:
                cfgsect, name, suffix = m.group(0), m.group(1), m.group(2)

                # Instantiate
                plugin = get_plugin(name, self, cfgsect, suffix)

                # Register as an event handler
                self.completed_step_handlers.append(plugin)

        # Delete the memory-intensive elements map from the system
        del self.system.ele_map

    def collect_stats(self, stats):
        super().collect_stats(stats)

        wtime = time.time() - self._wstart
        stats.set('solver-time-integrator', 'wall-time', wtime)

    def _accept_step(self, dt, idxcurr, err=None):
        self.tcurr += dt
        self.nacptsteps += 1
        self.nacptchain += 1
        self.stepinfo.append((dt, 'accept', err))

        self._idxcurr = idxcurr

        # Filter
        if self._fnsteps and self.nacptsteps % self._fnsteps == 0:
            self.system.filt(idxcurr)

        # Invalidate the solution cache
        self._curr_soln = None

        # Fire off any event handlers
        self.completed_step_handlers(self)

        # Clear the step info
        self.stepinfo = []

    def _reject_step(self, dt, idxold, err=None):
        if dt <= self.dtmin:
            raise RuntimeError('Minimum sized time step rejected')

        self.nacptchain = 0
        self.nrjctsteps += 1
        self.stepinfo.append((dt, 'reject', err))

        self._idxcurr = idxold

    @property
    def nsteps(self):
        return self.nacptsteps + self.nrjctsteps

    @property
    def soln(self):
        # If we do not have the solution cached then fetch it
        if not self._curr_soln:
            self._curr_soln = self.system.ele_scal_upts(self._idxcurr)

        return self._curr_soln


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
            dt = max(min(t - self.tcurr, self._dt), self.dtmin)

            # Take the step
            idxcurr = self.step(self.tcurr, dt)

            # We are not adaptive, so accept every step
            self._accept_step(dt, idxcurr)


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
        comm, rank, root = get_comm_rank_root()

        errest = self._get_errest_kerns()

        # Obtain an estimate for the error
        self._prepare_reg_banks(x, y, z)
        self._queue % errest(self._atol, self._rtol)

        # Reduce locally (element types) and globally (MPI ranks)
        rl = sum(errest.retval)
        rg = comm.allreduce(rl, op=get_mpi('sum'))

        # Normalise
        err = math.sqrt(rg / self._gndofs)

        return err if not math.isnan(err) else 100

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        # Constants
        maxf = self._maxfac
        minf = self._minfac
        saff = self._saffac
        sord = self._stepper_order

        expa = self._alpha / self._stepper_order
        expb = self._beta / self._stepper_order

        while self.tcurr < t:
            # Decide on the time step
            dt = max(min(t - self.tcurr, self._dt), self.dtmin)

            # Take the step
            idxcurr, idxprev, idxerr = self.step(self.tcurr, dt)

            # Estimate the error
            err = self._errest(idxerr, idxcurr, idxprev)

            # Determine time step adjustment factor
            fac = err**-expa * self._errprev**expb
            fac = min(maxf, max(minf, saff*fac))

            # Compute the size of the next step
            self._dt = fac*dt

            # Decide if to accept or reject the step
            if err < 1.0:
                self._errprev = err
                self._accept_step(dt, idxcurr, err=err)
            else:
                self._reject_step(dt, idxprev, err=err)
