from pyfr.mpiutil import get_comm_rank_root
from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)


class BaseAdvectionDiffusionIntInters(BaseAdvectionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, rhs, elemap, cfg)

        # Generate the additional view matrices
        self._vect_lhs = self._vect_view(lhs, 'get_vect_fpts_for_inter')
        self._vect_rhs = self._vect_view(rhs, 'get_vect_fpts_for_inter')
        self._comm_lhs = self._scal_view(lhs, 'get_comm_fpts_for_inter')
        self._comm_rhs = self._scal_view(rhs, 'get_comm_fpts_for_inter')

        # Generate the additional view matrices for artificial viscosity
        if cfg.get('solver', 'shock-capturing') == 'artificial-viscosity':
            self._artvisc_lhs = self._view(lhs, 'get_artvisc_fpts_for_inter')
            self._artvisc_rhs = self._view(rhs, 'get_artvisc_fpts_for_inter')
        else:
            self._artvisc_lhs = self._artvisc_rhs = None

        # Additional kernel constants
        self.c |= cfg.items_as('solver-interfaces', float)

    def _gen_perm(self, lhs, rhs):
        # In the special case of β = -0.5 it is better to sort by the
        # RHS interface; otherwise we simply opt for the LHS
        beta = self.cfg.getfloat('solver-interfaces', 'ldg-beta')
        side = lhs if beta != -0.5 else rhs

        # Compute the relevant permutation
        self._perm = self._get_perm_for_view(side, 'get_scal_fpts_for_inter')


class BaseAdvectionDiffusionMPIInters(BaseAdvectionMPIInters):
    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super().__init__(be, lhs, rhsrank, elemap, cfg)

        comm, rank, root = get_comm_rank_root()

        lhsprank = rank
        rhsprank = rhsrank

        # Generate second set of view matrices
        self._vect_lhs = self._vect_xchg_view(lhs, 'get_vect_fpts_for_inter')
        self._vect_rhs = be.xchg_matrix_for_view(self._vect_lhs)
        self._comm_lhs = self._scal_xchg_view(lhs, 'get_comm_fpts_for_inter')
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
        if (lhsprank + rhsprank) % 2:
            self.c['ldg-beta'] *= 1.0 if lhsprank > rhsprank else -1.0
        else:
            self.c['ldg-beta'] *= 1.0 if rhsprank > lhsprank else -1.0

        # Allocate a tag
        vect_fpts_tag = next(self._mpi_tag_counter)

        # If we need to send our gradients to the RHS
        if self.c['ldg-beta'] != -0.5:
            self.kernels['vect_fpts_pack'] = lambda: be.kernel(
                'pack', self._vect_lhs
            )
            self.mpireqs['vect_fpts_send'] = lambda: self._vect_lhs.sendreq(
                self._rhsrank, vect_fpts_tag
            )

        # If we need to recv gradients from the RHS
        if self.c['ldg-beta'] != 0.5:
            self.mpireqs['vect_fpts_recv'] = lambda: self._vect_rhs.recvreq(
                self._rhsrank, vect_fpts_tag
            )
            self.kernels['vect_fpts_unpack'] = lambda: be.kernel(
                'unpack', self._vect_rhs
            )

        # Generate the additional kernels/views for artificial viscosity
        if cfg.get('solver', 'shock-capturing') == 'artificial-viscosity':
            self._artvisc_lhs = self._xchg_view(lhs,
                                                'get_artvisc_fpts_for_inter')
            self._artvisc_rhs = be.xchg_matrix_for_view(self._artvisc_lhs)

            # Allocate a tag
            artvisc_fpts_tag = next(self._mpi_tag_counter)

            # If we need to send our artificial viscosity to the RHS
            if self.c['ldg-beta'] != -0.5:
                av_lhs = self._artvisc_lhs
                self.kernels['artvisc_fpts_pack'] = lambda: be.kernel(
                    'pack', av_lhs
                )
                self.mpireqs['artvisc_fpts_send'] = lambda: av_lhs.sendreq(
                    self._rhsrank, artvisc_fpts_tag
                )

            # If we need to recv artificial viscosity from the RHS
            if self.c['ldg-beta'] != 0.5:
                av_rhs = self._artvisc_rhs
                self.mpireqs['artvisc_fpts_recv'] = lambda: av_rhs.recvreq(
                    self._rhsrank, artvisc_fpts_tag
                )
                self.kernels['artvisc_fpts_unpack'] = lambda: be.kernel(
                    'unpack', av_rhs
                )
        else:
            self._artvisc_lhs = self._artvisc_rhs = None


class BaseAdvectionDiffusionBCInters(BaseAdvectionBCInters):
    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        # Additional view matrices
        self._vect_lhs = self._vect_view(lhs, 'get_vect_fpts_for_inter')
        self._comm_lhs = self._scal_view(lhs, 'get_comm_fpts_for_inter')

        # Additional kernel constants
        self.c |= cfg.items_as('solver-interfaces', float)

        # Generate the additional view matrices for artificial viscosity
        if cfg.get('solver', 'shock-capturing') == 'artificial-viscosity':
            self._artvisc_lhs = self._view(lhs, 'get_artvisc_fpts_for_inter')
        else:
            self._artvisc_lhs = None
