# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseInters, get_opt_view_perm
from pyfr.nputil import npeval


class BaseAdvectionIntInters(BaseInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super(BaseAdvectionIntInters, self).__init__(be, lhs, elemap, cfg)

        const_mat = self._const_mat

        # Compute the `optimal' permutation for our interface
        self._gen_perm(lhs, rhs)

        # Generate the left and right hand side view matrices
        self._scal0_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._scal0_rhs = self._scal_view(rhs, 'get_scal_fpts_for_inter')

        # Generate the constant matrices
        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._mag_pnorm_rhs = const_mat(rhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

    def _gen_perm(self, lhs, rhs):
        # Arbitrarily, take the permutation which results in an optimal
        # memory access pattern for the LHS of the interface
        self._perm = get_opt_view_perm(lhs, 'get_scal_fpts_for_inter',
                                       self._elemap)


class BaseAdvectionMPIInters(BaseInters):
    # Tag used for MPI
    MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        super(BaseAdvectionMPIInters, self).__init__(be, lhs, elemap, cfg)
        self._rhsrank = rhsrank
        self._rallocs = rallocs

        const_mat = self._const_mat

        # Generate the left hand view matrix and its dual
        self._scal0_lhs = self._scal_mpi_view(lhs, 'get_scal_fpts_for_inter')
        self._scal0_rhs = be.mpi_matrix_for_view(self._scal0_lhs)

        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

        # Kernels
        self.kernels['scal_fpts_pack'] = lambda: be.kernel(
            'pack', self._scal0_lhs
        )
        self.kernels['scal_fpts_send'] = lambda: be.kernel(
            'send_pack', self._scal0_lhs, self._rhsrank, self.MPI_TAG
        )
        self.kernels['scal_fpts_recv'] = lambda: be.kernel(
            'recv_pack', self._scal0_rhs, self._rhsrank, self.MPI_TAG
        )
        self.kernels['scal_fpts_unpack'] = lambda: be.kernel(
            'unpack', self._scal0_rhs
        )


class BaseAdvectionBCInters(BaseInters):
    type = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super(BaseAdvectionBCInters, self).__init__(be, lhs, elemap, cfg)
        self._cfgsect = cfgsect

        const_mat = self._const_mat

        # For BC interfaces, which only have an LHS state, we take the
        # permutation which results in an optimal memory access pattern
        # iterating over this state.
        self._perm = get_opt_view_perm(lhs, 'get_scal_fpts_for_inter', elemap)

        # LHS view and constant matrices
        self._scal0_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

    def _eval_opts(self, opts, default=None):
        # Boundary conditions, much like initial conditions, can be
        # parameterized by values in [constants] so we must bring these
        # into scope when evaluating the boundary conditions
        cc = self._cfg.items_as('constants', float)

        cfg, sect = self._cfg, self._cfgsect

        # Evaluate any BC specific arguments from the config file
        if default is not None:
            return [npeval(cfg.get(sect, k, default), cc) for k in opts]
        else:
            return [npeval(cfg.get(sect, k), cc) for k in opts]
