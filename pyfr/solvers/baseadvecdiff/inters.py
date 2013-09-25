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

        # Additional kernel constants
        self._tpl_c.update(cfg.items_as('solver-interfaces', float))

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

        # Generate second set of view matrices
        self._scal1_lhs = self._view_onto(lhs, 'get_scal_fpts1_for_inter')
        self._vect0_lhs = self._mpi_view_onto(lhs, 'get_vect_fpts0_for_inter')
        self._vect0_rhs = be.mpi_matrix_for_view(self._vect0_lhs)

        # Additional kernel constants
        self._tpl_c.update(cfg.items_as('solver-interfaces', float))

        # We require cflux(l,r,n_l) = -cflux(r,l,n_r) and
        # conu(l,r) = conu(r,l) and where l and r are left and right
        # solutions at an interface and n_[l,r] are physical normals.
        # The simplest way to enforce this at an MPI interface is for
        # one side to take β = -β for the cflux and conu kernels. We
        # pick this side (arbitrarily) by comparing the physical ranks
        # of the two partitions.
        self._tpl_c['ldg-beta'] *= 1.0 if lhsprank > rhsprank else -1.0


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


class BaseAdvectionDiffusionBCInters(BaseAdvectionBCInters):
    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super(BaseAdvectionDiffusionBCInters, self).__init__(be, lhs, elemap,
                                                             cfgsect, cfg)

        # Additional view matrices
        self._scal1_lhs = self._view_onto(lhs, 'get_scal_fpts1_for_inter')
        self._vect0_lhs = self._view_onto(lhs, 'get_vect_fpts0_for_inter')

        # Additional kernel constants
        self._tpl_c.update(cfg.items_as('solver-interfaces', float))
