# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from ConfigParser import NoOptionError

import numpy as np

from pyfr.nputil import npeval


def get_view_mats(interside, mat, elemap):
    # Map from element type to view mat getter
    viewmatmap = {type: getattr(ele, mat) for type, ele in elemap.items()}

    scal = []
    for type, eidx, face, rtag in interside:
        # After the += the length is increased by *three*
        scal += viewmatmap[type](eidx, face, rtag)

    # Concat the various numpy arrays together to yield the three matrices
    # required in order to define a view
    scal_v = [np.hstack(scal[i::3]) for i in xrange(3)]

    return scal_v


def get_mat(interside, mat, elemap):
    # Map from element type to view mat getter
    emap = {type: getattr(ele, mat) for type, ele in elemap.items()}

    # Form the matrix
    m = [emap[type](eidx, fidx, rtag) for type, eidx, fidx, rtag in interside]
    return np.concatenate(m)[None,...]


class BaseInters(object):
    __metaclass__ = ABCMeta

    def __init__(self, be, elemap, cfg):
        self._be = be
        self._elemap = elemap
        self._cfg = cfg

        # Get the number of dimensions and variables
        self.ndims = next(iter(elemap.viewvalues())).ndims
        self.nvars = next(iter(elemap.viewvalues())).nvars

    def _const_mat(self, inter, meth):
        m = get_mat(inter, meth, self._elemap)
        return self._be.const_matrix(m)

    def _view_onto(self, inter, meth):
        vm = get_view_mats(inter, meth, self._elemap)
        return self._be.view(*vm, vlen=self.nvars)

    def _mpi_view_onto(self, inter, meth):
        vm = get_view_mats(inter, meth, self._elemap)
        return self._be.mpi_view(*vm, vlen=self.nvars)

    def _kernel_constants(self):
        return self._cfg.items_as('constants', float)

    @abstractmethod
    def get_rsolve_kern(self):
        pass


class BaseAdvectionIntInters(BaseInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super(BaseAdvectionIntInters, self).__init__(be, elemap, cfg)

        view_onto, const_mat = self._view_onto, self._const_mat

        # Generate the left and right hand side view matrices
        self._scal0_lhs = view_onto(lhs, 'get_scal_fpts0_for_inter')
        self._scal0_rhs = view_onto(rhs, 'get_scal_fpts0_for_inter')

        # Generate the constant matrices
        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._mag_pnorm_rhs = const_mat(rhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')


class BaseAdvectionMPIInters(BaseInters):
    # Tag used for MPI
    MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        super(BaseAdvectionMPIInters, self).__init__(be, elemap, cfg)
        self._rhsrank = rhsrank
        self._rallocs = rallocs

        mpi_view_onto, const_mat = self._mpi_view_onto, self._const_mat

        # Generate the left hand view matrix and its dual
        self._scal0_lhs = mpi_view_onto(lhs, 'get_scal_fpts0_for_inter')
        self._scal0_rhs = be.mpi_matrix_for_view(self._scal0_lhs)

        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

    def get_scal_fpts0_pack_kern(self):
        return self._be.kernel('pack', self._scal0_lhs)

    def get_scal_fpts0_send_pack_kern(self):
        return self._be.kernel('send_pack', self._scal0_lhs,
                               self._rhsrank, self.MPI_TAG)

    def get_scal_fpts0_recv_pack_kern(self):
        return self._be.kernel('recv_pack', self._scal0_rhs,
                               self._rhsrank, self.MPI_TAG)

    def get_scal_fpts0_unpack_kern(self):
        return self._be.kernel('unpack', self._scal0_rhs)


class BaseAdvectionDiffusionIntInters(BaseAdvectionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        base = super(BaseAdvectionDiffusionIntInters, self)
        base.__init__(be, lhs, rhs, elemap, cfg)

        # Generate the additional view matrices
        self._scal1_lhs = self._view_onto(lhs, 'get_scal_fpts1_for_inter')
        self._scal1_rhs = self._view_onto(rhs, 'get_scal_fpts1_for_inter')
        self._vect0_lhs = self._view_onto(lhs, 'get_vect_fpts0_for_inter')
        self._vect0_rhs = self._view_onto(rhs, 'get_vect_fpts0_for_inter')

    def _kernel_constants(self):
        kc = super(BaseAdvectionDiffusionIntInters, self)._kernel_constants()

        # Bring LDG-specific constants into scope
        kc.update(self._cfg.items_as('solver-interfaces', float))

        return kc

    def get_conu_fpts_kern(self):
        kc = self._kernel_constants()
        return self._be.kernel('conu_int', self.nvars,
                               self._scal0_lhs, self._scal0_rhs,
                               self._scal1_lhs, self._scal1_rhs, kc)


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

    def _kernel_constants(self):
        kc = super(BaseAdvectionDiffusionMPIInters, self)._kernel_constants()

        # Bring LDG-specific constants into scope
        kc.update(self._cfg.items_as('solver-interfaces', float))
        kc['ldg-beta'] *= self._beta_sgn

        return kc

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
        kc = self._kernel_constants()
        return self._be.kernel('conu_mpi', self.nvars,
                               self._scal0_lhs, self._scal0_rhs,
                               self._scal1_lhs, kc)


class BaseAdvectionBCInters(BaseInters):
    type = None
    args = []

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super(BaseAdvectionBCInters, self).__init__(be, elemap, cfg)
        self._cfgsect = cfgsect

        view_onto, const_mat = self._view_onto, self._const_mat

        # LHS view and constant matrices
        self._scal0_lhs = view_onto(lhs, 'get_scal_fpts0_for_inter')
        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

    def _kernel_constants(self):
        kc = super(BaseAdvectionBCInters, self)._kernel_constants()

        # Boundary conditions, much like initial conditions, can be
        # parameterized by values in [constants] so we must bring these
        # into scope when evaluating the boundary conditions
        cc = self._cfg.items_as('constants', float)

        # Evaluate any BC specific arguments from the config file
        for k in self.args:
            try:
                # Get the constant/expression
                expr = self._cfg.get(self._cfgsect, k)

                # Evaluate
                kc[k] = npeval(expr, cc)
            except NoOptionError:
                continue

        return kc


class BaseAdvectionDiffusionBCInters(BaseAdvectionBCInters):
    def __init__(self, be, lhs, elemap, cfg):
        super(BaseAdvectionDiffusionBCInters, self).__init__(be, lhs,
                                                             elemap, cfg)

        # Additional view matrices
        self._scal1_lhs = self._view_onto(lhs, 'get_scal_fpts1_for_inter')
        self._vect0_lhs = self._view_onto(lhs, 'get_vect_fpts0_for_inter')


    def _kernel_constants(self):
        kc = super(BaseAdvectionDiffusionBCInters, self)._kernel_constants()

        # Bring LDG-specific constants into scope
        kc.update(self._cfg.items_as('solver-interfaces', float))

        return kc

    def get_conu_fpts_kern(self):
        kc = self._kernel_constants()
        return self._be.kernel('conu_bc', self.ndims, self.nvars, self.type,
                               self._scal0_lhs, self._scal1_lhs, kc)


class EulerIntInters(BaseAdvectionIntInters):
    def get_rsolve_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants()

        return self._be.kernel('rsolve_inv_int', self.ndims, self.nvars,
                               rsinv, self._scal0_lhs, self._scal0_rhs,
                               self._mag_pnorm_lhs, self._mag_pnorm_rhs,
                               self._norm_pnorm_lhs, kc)


class EulerMPIInters(BaseAdvectionMPIInters):
    def get_rsolve_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants()

        return self._be.kernel('rsolve_inv_mpi', self.ndims, self.nvars,
                               rsinv, self._scal0_lhs, self._scal0_rhs,
                               self._mag_pnorm_lhs, self._norm_pnorm_lhs, kc)


class EulerBaseBCInters(BaseAdvectionBCInters):
    def get_rsolve_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants()

        return self._be.kernel('rsolve_inv_bc', self.ndims, self.nvars,
                               rsinv, self.type, self._scal0_lhs,
                               self._mag_pnorm_lhs, self._norm_pnorm_lhs, kc)


class EulerSupInflowBCInters(EulerBaseBCInters):
    type = 'sup-inflow'
    args = ['fs-rho', 'fs-p', 'fs-u', 'fs-v', 'fs-w']


class NavierStokesIntInters(BaseAdvectionDiffusionIntInters):
    def get_rsolve_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants()

        return self._be.kernel('rsolve_ldg_vis_int', self.ndims, self.nvars,
                               rsinv, self._scal0_lhs, self._vect0_lhs,
                               self._scal0_rhs, self._vect0_rhs,
                               self._mag_pnorm_lhs, self._mag_pnorm_rhs,
                               self._norm_pnorm_lhs, kc)


class NavierStokesMPIInters(BaseAdvectionDiffusionMPIInters):
    def get_rsolve_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants()

        return self._be.kernel('rsolve_ldg_vis_mpi', self.ndims, self.nvars,
                               rsinv, self._scal0_lhs, self._vect0_lhs,
                               self._scal0_rhs, self._vect0_rhs,
                               self._mag_pnorm_lhs, self._norm_pnorm_lhs, kc)


class NavierStokesBaseBCInters(BaseAdvectionDiffusionBCInters):
    def get_rsolve_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants()

        return self._be.kernel('rsolve_ldg_vis_bc', self.ndims, self.nvars,
                               rsinv, self.type, self._scal0_lhs,
                               self._vect0_lhs, self._mag_pnorm_lhs,
                               self._norm_pnorm_lhs, kc)


class NavierStokesIsoThermNoslipBCInters(NavierStokesBaseBCInters):
    type = 'isotherm-noslip'
    args = ['cpTw']


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-inflow'
    args = ['fs-rho', 'fs-p', 'fs-u', 'fs-v', 'fs-w']


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-outflow'


class NavierStokesSubInflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-inflow'
    args = ['fs-rho', 'fs-p', 'fs-u', 'fs-v', 'fs-w']


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-outflow'
    args = ['fs-p']
