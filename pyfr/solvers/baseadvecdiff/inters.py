from pyfr.mpiutil import get_comm_rank_root
from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)


class BaseAdvectionDiffusionIntInters(BaseAdvectionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, rhs, elemap, cfg)

        # Generate the additional view matrices
        self._vect_lhs = self._vect_view(lhs, 'get_vect_fpts_for_inters')
        self._vect_rhs = self._vect_view(rhs, 'get_vect_fpts_for_inters')
        self._comm_lhs = self._scal_view(lhs, 'get_comm_fpts_for_inters')
        self._comm_rhs = self._scal_view(rhs, 'get_comm_fpts_for_inters')

        # Artificial viscosity (populated by ArtificialViscosity if active)
        self.artvisc = None

        # Additional kernel constants
        self.c |= cfg.items_as('solver-interfaces', float)

    def _gen_perm(self, lhs, rhs, scal):
        # In the special case of β = -0.5 it is better to sort by the
        # RHS interface; otherwise we simply opt for the LHS
        beta = self.cfg.getfloat('solver-interfaces', 'ldg-beta')
        side = lhs if beta != -0.5 else rhs

        # Compute the relevant permutation
        self._perm = self._get_perm_for_field(side, scal)


class BaseAdvectionDiffusionMPIInters(BaseAdvectionMPIInters):
    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super().__init__(be, lhs, rhsrank, elemap, cfg)

        comm, rank, root = get_comm_rank_root()

        # Generate second set of view matrices
        self._vect_lhs = self._vect_xchg_view(lhs, 'get_vect_fpts_for_inters')
        self._vect_rhs = be.xchg_matrix_for_view(self._vect_lhs)
        self._comm_lhs = self._scal_xchg_view(lhs, 'get_comm_fpts_for_inters')
        self._comm_rhs = be.xchg_matrix_for_view(self._comm_lhs)

        # Additional kernel constants
        self.c |= cfg.items_as('solver-interfaces', float)

        # We require cflux(l,r,n_l) = -cflux(r,l,n_r) and
        # conu(l,r) = conu(r,l) and where l and r are left and right
        # solutions at an interface and n_[l,r] are physical normals.
        # The simplest way to enforce this at an MPI interface is for
        # one side to take β = -β for the cflux and conu kernels. We
        # pick this side (arbitrarily) by comparing the physical ranks
        # of the two partitions.
        if (rank + rhsrank) % 2:
            self.c['ldg-beta'] *= 1.0 if rank > rhsrank else -1.0
        else:
            self.c['ldg-beta'] *= 1.0 if rhsrank > rank else -1.0

        # Artificial viscosity (populated by ArtificialViscosity if active)
        self.artvisc = None


class BaseAdvectionDiffusionBCInters(BaseAdvectionBCInters):
    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # Additional view matrices
        self._vect_lhs = self._vect_view(lhs, 'get_vect_fpts_for_inters')
        self._comm_lhs = self._scal_view(lhs, 'get_comm_fpts_for_inters')

        # Additional kernel constants
        self.c |= cfg.items_as('solver-interfaces', float)

        # Artificial viscosity (populated by ArtificialViscosity if active)
        self.artvisc = None
