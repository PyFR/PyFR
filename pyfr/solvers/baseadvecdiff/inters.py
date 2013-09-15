# -*- coding: utf-8 -*-

from pyfr.solvers.base import get_opt_view_perm
from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)


class BaseAdvectionDiffusionIntInters(BaseAdvectionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        base = super(BaseAdvectionDiffusionIntInters, self)
        base.__init__(be, lhs, rhs, elemap, cfg)

        # Generate the additional view matrices
        self._scal1_lhs = self._view_onto(lhs, 'get_scal_fpts1_for_inter')
        self._scal1_rhs = self._view_onto(rhs, 'get_scal_fpts1_for_inter')
        self._vect0_lhs = self._view_onto(lhs, 'get_vect_fpts0_for_inter')
        self._vect0_rhs = self._view_onto(rhs, 'get_vect_fpts0_for_inter')

    @property
    def _kernel_constants(self):
        kc = super(BaseAdvectionDiffusionIntInters, self)._kernel_constants

        # Bring LDG-specific constants into scope
        newkc = dict(kc)
        newkc.update(self._cfg.items_as('solver-interfaces', float))

        return newkc

    def get_conu_fpts_kern(self):
        kc = self._kernel_constants
        return self._be.kernel('conu_int', self.nvars,
                               self._scal0_lhs, self._scal0_rhs,
                               self._scal1_lhs, self._scal1_rhs, kc)

    def _gen_perm(self, lhs, rhs):
        # In the special case of β = -0.5 it is better to sort by the
        # RHS interface; otherwise we simply opt for the LHS
        beta = self._cfg.getfloat('solver-interfaces', 'ldg-beta')
        side = lhs if beta != -0.5 else rhs

        # Compute the relevant permutation
        self._perm = get_opt_view_perm(side, 'get_scal_fpts0_for_inter',
                                       self._elemap)


class BaseAdvectionDiffusionMPIInters(BaseAdvectionMPIInters):
    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        base = super(BaseAdvectionDiffusionMPIInters, self)
        base.__init__(be, lhs, rhsrank, rallocs, elemap, cfg)

        lhsprank = rallocs.prank
        rhsprank = rallocs.mprankmap[rhsrank]

        # We require rsolve(l,r,n_l) = -rsolve(r,l,n_r) and
        # conu(l,r) = conu(r,l) and where l and r are left and right
        # solutions at an interface and n_[l,r] are physical normals.
        # The simplest way to enforce this at an MPI interface is for
        # one side to take β = -β for the rsolve and conu kernels. We
        # pick this side (arbitrarily) by comparing the physical ranks
        # of the two partitions.
        self._beta_sgn = 1.0 if lhsprank > rhsprank else -1.0

        # Generate second set of view matrices
        self._scal1_lhs = self._view_onto(lhs, 'get_scal_fpts1_for_inter')
        self._vect0_lhs = self._mpi_view_onto(lhs, 'get_vect_fpts0_for_inter')
        self._vect0_rhs = be.mpi_matrix_for_view(self._vect0_lhs)

    @property
    def _kernel_constants(self):
        kc = super(BaseAdvectionDiffusionMPIInters, self)._kernel_constants

        # Bring LDG-specific constants into scope
        newkc = dict(kc)
        newkc.update(self._cfg.items_as('solver-interfaces', float))
        newkc['ldg-beta'] *= self._beta_sgn

        return newkc

    def get_vect_fpts0_pack_kern(self):
        return self._be.kernel('pack', self._vect0_lhs)

    def get_vect_fpts0_send_pack_kern(self):
        return self._be.kernel('send_pack', self._vect0_lhs,
                               self._rhsrank, self.MPI_TAG)

    def get_vect_fpts0_recv_pack_kern(self):
        return self._be.kernel('recv_pack', self._vect0_rhs,
                               self._rhsrank, self.MPI_TAG)

    def get_vect_fpts0_unpack_kern(self):
        return self._be.kernel('unpack', self._vect0_rhs)

    def get_conu_fpts_kern(self):
        kc = self._kernel_constants
        return self._be.kernel('conu_mpi', self.nvars,
                               self._scal0_lhs, self._scal0_rhs,
                               self._scal1_lhs, kc)


class BaseAdvectionDiffusionBCInters(BaseAdvectionBCInters):
    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super(BaseAdvectionDiffusionBCInters, self).__init__(be, lhs, elemap,
                                                             cfgsect, cfg)

        # Additional view matrices
        self._scal1_lhs = self._view_onto(lhs, 'get_scal_fpts1_for_inter')
        self._vect0_lhs = self._view_onto(lhs, 'get_vect_fpts0_for_inter')

    @property
    def _kernel_constants(self):
        kc = super(BaseAdvectionDiffusionBCInters, self)._kernel_constants

        # Bring LDG-specific constants into scope
        newkc = dict(kc)
        newkc.update(self._cfg.items_as('solver-interfaces', float))

        return newkc

    def get_conu_fpts_kern(self):
        kc = self._kernel_constants
        return self._be.kernel('conu_bc', self.ndims, self.nvars, self.type,
                               self._scal0_lhs, self._scal1_lhs, kc)
