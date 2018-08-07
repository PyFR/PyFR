# -*- coding: utf-8 -*-

import numpy as np

from pyfr.integrators.dual.pseudo.base import BaseDualPseudoIntegrator
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.util import memoize


class BaseDualPseudoController(BaseDualPseudoIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pseudo-step counter
        self.npseudosteps = 0

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Stats on the most recent step
        self.pseudostepinfo = []

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


class DualPseudoNoneController(BaseDualPseudoController):
    pseudo_controller_name = 'none'

    @property
    def _controller_needs_lerrest(self):
        return False

    def conv_mon(self, i, minniters):
        if i >= minniters - 1:
            # Subtract the current solution from the previous solution
            self._add(-1.0, self._idxprev, 1.0, self._idxcurr)

            # Compute the normalised residual
            resid = tuple(self._resid(self.dtau, self._idxprev))
        else:
            resid = None

        # Increment the step count
        self.npseudosteps += 1

        self.pseudostepinfo.append((self.npseudosteps, i + 1, resid))

        return resid and max(resid[1:]) < self._pseudo_residtol

    def pseudo_advance(self, tcurr, tout, dt):
        self.tcurr = tcurr
        self.dtau = max(min(tout - tcurr, self._dtau), self._dtaumin)

        for i in range(self.maxniters):
            # Take the step
            self._idxcurr, self._idxprev = self.step(self.tcurr, dt, self.dtau)

            # Convergence monitoring
            if self.conv_mon(i, self.minniters):
                break

        # Update
        self.finalise_pseudo_advance(self._idxcurr)
