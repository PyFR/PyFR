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


class DualNonePseudoController(BaseDualPseudoController):
    pseudo_controller_name = 'none'

    @property
    def _pseudo_controller_needs_lerrest(self):
        return False

    def convmon(self, i, minniters):
        # Increment the step count
        self.npseudosteps += 1

        if i >= minniters - 1:
            # Subtract the current solution from the previous solution
            self._add(-1.0, self._idxprev, 1.0, self._idxcurr)

            # Compute the normalised residual
            resid = tuple(self._resid(self._dtau, self._idxprev))

            self.pseudostepinfo.append((self.npseudosteps, i + 1, resid))
            return all(r <= t for r, t in zip(resid, self._pseudo_residtol))
        else:
            self.pseudostepinfo.append((self.npseudosteps, i + 1, None))
            return False

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        for i in range(self.maxniters):
            # Take the step
            self._idxcurr, self._idxprev = self.step(self.tcurr)

            # Convergence monitoring
            if self.convmon(i, self.minniters):
                break

        # Update
        self.finalise_pseudo_advance(self._idxcurr)


class DualPIPseudoController(BaseDualPseudoController):
    pseudo_controller_name = 'local-pi'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sect = 'solver-time-integrator'

        # Error norm
        self._norm = self.cfg.get(sect, 'errest-norm', 'l2')
        if self._norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid error norm')

        tplargs = {'ndims': self.system.ndims, 'nvars': self.system.nvars}

        # Error tolerance
        tplargs['atol'] = self.cfg.getfloat(sect, 'atol')

        # PI control values
        sord = self._pseudo_stepper_order
        tplargs['expa'] = self.cfg.getfloat(sect, 'pi-alpha', 0.7) / sord
        tplargs['expb'] = self.cfg.getfloat(sect, 'pi-beta', 0.4) / sord

        # Constants
        tplargs['maxf'] = self.cfg.getfloat(sect, 'max-fact', 1.01)
        tplargs['minf'] = self.cfg.getfloat(sect, 'min-fact', 0.98)
        tplargs['saff'] = self.cfg.getfloat(sect, 'safety-fact', 0.8)
        tplargs['dtau_maxf'] = self.cfg.getfloat(sect, 'pseudo-dt-max-mult',
                                                 3.0)

        # Limits for the local pseudo-time-step size
        tplargs['dtau_min'] = self._dtau
        tplargs['dtau_max'] = tplargs['dtau_maxf'] * self._dtau

        # Register a kernel to compute local error
        self.backend.pointwise.register(
            'pyfr.integrators.dual.pseudo.kernels.localerrest'
        )

        for ele, shape, dtaumat in zip(self.system.ele_map.values(),
                                       self.system.ele_shapes, self.dtau_upts):
            # Allocate storage for previous error
            err_prev = self.backend.matrix(shape, np.ones(shape),
                                           tags={'align'})

            # Append the error kernels to the proxylist
            self.pintgkernels['localerrest'].append(
                self.backend.kernel(
                    'localerrest', tplargs=tplargs,
                    dims=[ele.nupts, ele.neles], err=ele.scal_upts_inb,
                    errprev=err_prev, dtau_upts=dtaumat
                )
            )

        self.backend.commit()

    @property
    def _pseudo_controller_needs_lerrest(self):
        return True

    def localerrest(self, errbank):
        self.system.eles_scal_upts_inb.active = errbank
        self._queue % self.pintgkernels['localerrest']()

    def convmon(self, i, minniters):
        # Increment the step count
        self.npseudosteps += 1

        if i >= minniters - 1:
            # Subtract the current solution from the previous solution
            self._add(-1.0, self._idxprev, 1.0, self._idxcurr)

            # Divide by 1/dtau
            self.localdtau(self._idxprev, inv=1)

            # Reduction
            resid = tuple(self._resid(1.0, self._idxprev))

            self.pseudostepinfo.append((self.npseudosteps, i + 1, resid))
            return all(r <= t for r, t in zip(resid, self._pseudo_residtol))
        else:
            self.pseudostepinfo.append((self.npseudosteps, i + 1, None))
            return False

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        for i in range(self.maxniters):
            # Take the step
            self._idxcurr, self._idxprev, self._idxerr = self.step(self.tcurr)
            self.localerrest(self._idxerr)

            if self.convmon(i, self.minniters):
                break

        # Update
        self.finalise_pseudo_advance(self._idxcurr)
