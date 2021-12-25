# -*- coding: utf-8 -*-

import math

import numpy as np

from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.mpiutil import get_comm_rank_root, get_mpi


class BaseStdController(BaseStdIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Stats on the most recent step
        self.stepinfo = []

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            for csh in self.completed_step_handlers:
                csh(self)

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

        # Invalidate the solution gradients cache
        self._curr_grad_soln = None

        # Fire off any event handlers
        for csh in self.completed_step_handlers:
            csh(self)

        # Abort if plugins request it
        self._check_abort()

        # Clear the step info
        self.stepinfo = []

    def _reject_step(self, dt, idxold, err=None):
        if dt <= self.dtmin:
            raise RuntimeError('Minimum sized time step rejected')

        self.nacptchain = 0
        self.nrjctsteps += 1
        self.stepinfo.append((dt, 'reject', err))

        self._idxcurr = idxold


class StdNoneController(BaseStdController):
    controller_name = 'none'

    @property
    def controller_needs_errest(self):
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


class StdPIController(BaseStdController):
    controller_name = 'pi'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = 'solver-time-integrator'

        # Maximum time step
        self.dtmax = self.cfg.getfloat(sect, 'dt-max', 1e2)

        # Error tolerances
        self._atol = self.cfg.getfloat(sect, 'atol')
        self._rtol = self.cfg.getfloat(sect, 'rtol')

        # Error norm
        self._norm = self.cfg.get(sect, 'errest-norm', 'l2')
        if self._norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid error norm')

        # PI control values
        self._alpha = self.cfg.getfloat(sect, 'pi-alpha', 0.7)
        self._beta = self.cfg.getfloat(sect, 'pi-beta', 0.4)

        # Estimate of previous error
        self._errprev = 1.0

        # Step size adjustment factors
        self._saffac = self.cfg.getfloat(sect, 'safety-fact', 0.8)
        self._maxfac = self.cfg.getfloat(sect, 'max-fact', 2.5)
        self._minfac = self.cfg.getfloat(sect, 'min-fact', 0.3)

        if not self._minfac < 1 <= self._maxfac:
            raise ValueError('Invalid max-fact, min-fact')

    @property
    def controller_needs_errest(self):
        return True

    def _errest(self, rcurr, rprev, rerr):
        comm, rank, root = get_comm_rank_root()

        errest = self._get_reduction_kerns(rcurr, rprev, rerr, method='errest',
                                           norm=self._norm)

        # Obtain an estimate for the squared error
        self._queue.enqueue_and_run(errest, self._atol, self._rtol)

        # L2 norm
        if self._norm == 'l2':
            # Reduce locally (element types + field variables)
            err = np.array([sum(v for e in errest for v in e.retval)])

            # Reduce globally (MPI ranks)
            comm.Allreduce(get_mpi('in_place'), err, op=get_mpi('sum'))

            # Normalise
            err = math.sqrt(float(err) / self._gndofs)
        # L^∞ norm
        else:
            # Reduce locally (element types + field variables)
            err = np.array([max(v for e in errest for v in e.retval)])

            # Reduce globally (MPI ranks)
            comm.Allreduce(get_mpi('in_place'), err, op=get_mpi('max'))

            # Normalise
            err = math.sqrt(float(err))

        return err if not math.isnan(err) else 100

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        # Constants
        maxf = self._maxfac
        minf = self._minfac
        saff = self._saffac
        sord = self.stepper_order

        expa = self._alpha / sord
        expb = self._beta / sord

        while self.tcurr < t:
            # Decide on the time step
            dt = max(min(t - self.tcurr, self._dt, self.dtmax), self.dtmin)

            # Take the step
            idxcurr, idxprev, idxerr = self.step(self.tcurr, dt)

            # Estimate the error
            err = self._errest(idxcurr, idxprev, idxerr)

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
