# -*- coding: utf-8 -*-

import numpy as np

from pyfr.integrators.dual.base import BaseDualIntegrator
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.util import memoize


class BaseDualController(BaseDualIntegrator):
    def __init__(self, *args, **kwargs):
        # Pseudo-step counter
        self.npseudosteps = 0

        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Stats on the most recent step
        self.pseudostepinfo = []

    def _accept_step(self, dt, idxcurr):
        self.tcurr += dt
        self.nacptsteps += 1
        self.nacptchain += 1

        self._idxcurr = idxcurr

        # Filter
        if self._fnsteps and self.nacptsteps % self._fnsteps == 0:
            self.system.filt(idxcurr)

        # Invalidate the solution cache
        self._curr_soln = None

        # Fire off any event handlers
        self.completed_step_handlers(self)

        # Clear the pseudo step info
        self.pseudostepinfo = []


class DualNoneController(BaseDualController):
    controller_name = 'none'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = 'solver-time-integrator'

        self._maxniters = self.cfg.getint(sect, 'pseudo-niters-max')
        self._minniters = self.cfg.getint(sect, 'pseudo-niters-min')

        if self._maxniters < self._minniters:
            raise ValueError('The maximum number of pseudo-iterations must '
                             'be greater than or equal to the minimum')

        self._pseudo_residtol= self.cfg.getfloat(sect, 'pseudo-resid-tol')
        self._pseudo_norm = self.cfg.get(sect, 'pseudo-resid-norm', 'uniform')

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            for i in range(self._maxniters):
                dt = max(min(t - self.tcurr, self._dt), self.dtmin)
                dtau = max(min(t - self.tcurr, self._dtau), self.dtaumin)

                # Take the step
                self._idxcurr, self._idxprev = self.step(self.tcurr, dt, dtau)

                # Activate convergence monitoring after pseudo-niters-min
                if i >= self._minniters - 1:
                    # Subtract the current solution from the previous solution
                    self._add(-1.0, self._idxprev, 1.0, self._idxcurr)

                    # Compute the normalised residual and check for convergence
                    resid = self._resid(dtau, self._idxprev)
                    self.pseudostepinfo.append((self.npseudosteps + 1, i + 1,
                                                tuple(resid)))

                    if max(resid) < self._pseudo_residtol:
                        break
                else:
                    self.pseudostepinfo.append((self.npseudosteps + 1, i + 1,
                                                (None,)*self.system.nvars))

                self.npseudosteps += 1

            # Update the dual-time stepping banks (n+1 => n, n => n-1)
            self.finalise_step(self._idxcurr)

            # We are not adaptive, so accept every step
            self._accept_step(dt, self._idxcurr)

    @memoize
    def _get_errest_kerns(self):
        return self._get_kernels('errest', nargs=3, norm=self._pseudo_norm)

    def _resid(self, dtau, x):
        comm, rank, root = get_comm_rank_root()

        # Get an errest kern to compute the square of the maximum residual
        errest = self._get_errest_kerns()

        # Prepare and run the kernel
        self._prepare_reg_banks(x, x, x)
        self._queue % errest(dtau, 0.0)

        # L2 norm
        if self._pseudo_norm == 'l2':
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([sum(ev) for ev in zip(*errest.retval)])
            comm.Allreduce(get_mpi('in_place'), res, op=get_mpi('sum'))

            # Normalise and return
            return np.sqrt(res / self._gndofs)
        # L^∞ norm
        else:
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([max(ev) for ev in zip(*errest.retval)])
            comm.Allreduce(get_mpi('in_place'), res, op=get_mpi('max'))

            # Normalise and return
            return np.sqrt(res)
