# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

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

def get_mag_pnorm_mat(interside, elemap):
    mag_pnorms = [elemap[type].get_mag_pnorms_for_inter(eidx, fidx, rtag)
                  for type, eidx, fidx, rtag in interside]

    return np.concatenate(mag_pnorms)[None,...]

def get_norm_pnorm_mat(interside, elemap):
    norm_pnorms = [elemap[type].get_norm_pnorms_for_inter(eidx, fidx, rtag)
                   for type, eidx, fidx, rtag in interside]

    return np.concatenate(norm_pnorms)[None,...]

class BaseInterfaces(object):
    __metaclass__ = ABCMeta

    def __init__(self, be, elemap, cfg):
        self._be = be
        self._cfg = cfg

        # Get the number of dimensions and variables
        self.ndims = next(iter(elemap.viewvalues())).ndims
        self.nvars = next(iter(elemap.viewvalues())).nvars

    @abstractmethod
    def get_rsolve_kern(self):
        pass


class BaseInternalInterfaces(BaseInterfaces):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super(BaseInternalInterfaces, self).__init__(be, elemap, cfg)

        # Generate the left and right hand side view matrices
        scal0_lhs = get_view_mats(lhs, 'get_scal_fpts0_for_inter', elemap)
        scal0_rhs = get_view_mats(rhs, 'get_scal_fpts0_for_inter', elemap)

        # Allocate these on the backend as views
        self._scal0_lhs = be.view(*scal0_lhs, vlen=self.nvars, tags={'nopad'})
        self._scal0_rhs = be.view(*scal0_rhs, vlen=self.nvars, tags={'nopad'})

        # Get the left and right hand side physical normal magnitudes
        mag_pnorm_lhs = get_mag_pnorm_mat(lhs, elemap)
        mag_pnorm_rhs = get_mag_pnorm_mat(rhs, elemap)

        # Allocate as a const matrix
        self._mag_pnorm_lhs = be.const_matrix(mag_pnorm_lhs, tags={'nopad'})
        self._mag_pnorm_rhs = be.const_matrix(mag_pnorm_rhs, tags={'nopad'})

        # Get the left hand side normalized physical normals
        norm_pnorm_lhs = get_norm_pnorm_mat(lhs, elemap)

        # Allocate as a const matrix
        self._norm_pnorm_lhs = be.const_matrix(norm_pnorm_lhs, tags={'nopad'})


class BaseMPIInterfaces(BaseInterfaces):
    # Tag used for MPI
    MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        super(BaseMPIInterfaces, self).__init__(be, elemap, cfg)
        self._rhsrank = rhsrank
        self._rallocs = rallocs

        # Generate the left hand view matrices
        scal0_lhs = get_view_mats(lhs, 'get_scal_fpts0_for_inter', elemap)

        # Allocate on the backend
        self._scal0_lhs = be.mpi_view(*scal0_lhs, vlen=self.nvars,
                                      tags={'nopad'})
        self._scal0_rhs = be.mpi_matrix_for_view(self._scal0_lhs)

        # Get the left hand side physical normal data
        mag_pnorm_lhs = get_mag_pnorm_mat(lhs, elemap)
        norm_pnorm_lhs = get_norm_pnorm_mat(lhs, elemap)

        # Allocate
        self._mag_pnorm_lhs = be.const_matrix(mag_pnorm_lhs, tags={'nopad'})
        self._norm_pnorm_lhs = be.const_matrix(norm_pnorm_lhs, tags={'nopad'})


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


class EulerInternalInterfaces(BaseInternalInterfaces):
    def get_rsolve_kern(self):
        gamma = self._cfg.getfloat('constants', 'gamma')

        return self._be.kernel('rsolve_rus_inv_int', self.ndims, self.nvars,
                               self._scal0_lhs, self._scal0_rhs,
                               self._mag_pnorm_lhs, self._mag_pnorm_rhs,
                               self._norm_pnorm_lhs, gamma)


class EulerMPIInterfaces(BaseMPIInterfaces):
    def get_rsolve_kern(self):
        gamma = self._cfg.getfloat('constants', 'gamma')

        return self._be.kernel('rsolve_rus_inv_mpi', self.ndims, self.nvars,
                               self._scal0_lhs, self._scal0_rhs,
                               self._mag_pnorm_lhs, self._norm_pnorm_lhs,
                               gamma)


class NavierStokesInternalInterfaces(BaseInternalInterfaces):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super(NavierStokesInternalInterfaces, self).__init__(be, lhs, rhs,
                                                             elemap, cfg)

        # Generate the second set of scalar view matrices
        scal1_lhs = get_view_mats(lhs, 'get_scal_fpts1_for_inter', elemap)
        scal1_rhs = get_view_mats(rhs, 'get_scal_fpts1_for_inter', elemap)

        # And the vector view matrices
        vect0_lhs = get_view_mats(lhs, 'get_vect_fpts0_for_inter', elemap)
        vect0_rhs = get_view_mats(rhs, 'get_vect_fpts0_for_inter', elemap)

        # Allocate these on the backend as views
        self._scal1_lhs = be.view(*scal1_lhs, vlen=self.nvars, tags={'nopad'})
        self._scal1_rhs = be.view(*scal1_rhs, vlen=self.nvars, tags={'nopad'})
        self._vect0_lhs = be.view(*vect0_lhs, vlen=self.nvars, tags={'nopad'})
        self._vect0_rhs = be.view(*vect0_rhs, vlen=self.nvars, tags={'nopad'})

    def get_conu_fpts_kern(self):
        beta = self._cfg.getfloat('constants', 'beta')

        return self._be.kernel('conu_int', self.nvars,
                               self._scal0_lhs, self._scal0_rhs,
                               self._scal1_lhs, self._scal1_rhs, beta)

    def get_rsolve_kern(self):
        # Flux function constants
        gamma = self._cfg.getfloat('constants', 'gamma')
        mu = self._cfg.getfloat('constants', 'mu')
        pr = self._cfg.getfloat('constants', 'Pr')

        # Riemann solver constants
        beta = self._cfg.getfloat('constants', 'beta')
        tau = self._cfg.getfloat('constants', 'tau')

        return self._be.kernel('rsolve_ldg_vis_int', self.ndims, self.nvars,
                               self._scal0_lhs, self._vect0_lhs,
                               self._scal0_rhs, self._vect0_rhs,
                               self._mag_pnorm_lhs, self._mag_pnorm_rhs,
                               self._norm_pnorm_lhs, gamma, mu, pr, beta, tau)


class NavierStokesMPIInterfaces(BaseMPIInterfaces):
    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        super(NavierStokesMPIInterfaces, self).__init__(be, lhs, rhsrank,
                                                        rallocs, elemap, cfg)

        lhsprank = rallocs.prank
        rhsprank = rallocs.mprankmap[rhsrank]

        # We require rsolve(l,r,n_l) = -rsolve(r,l,n_r) and
        # conu(l,r) = conu(r,l) and where l and r are left and right
        # solutions at an interface and n_[l,r] are physical normals.
        # The simplest way to enforce this at an MPI interface is for
        # one side to take β = -β for the rsolve and conu kernels. We
        # pick this side (arbitrarily) by comparing the physical ranks
        # of the two partitions.
        self._sign = 1.0 if lhsprank > rhsprank else -1.0

        # Generate the second set of scalar view matrices
        scal1_lhs = get_view_mats(lhs, 'get_scal_fpts1_for_inter', elemap)
        scal1_rhs = get_view_mats(rhs, 'get_scal_fpts1_for_inter', elemap)

        self._scal1_lhs = be.view(*scal1_lhs, vlen=self.nvars, tags={'nopad'})
        self._scal1_rhs = be.view(*scal1_rhs, vlen=self.nvars, tags={'nopad'})

        vect0_lhs = get_vect_mats(lhs, 'get_vect_fpts0_for_inter', elemap)

        self._vect0_lhs = be.mpi_view(*vect0_lhs, vlen=self.nvars,
                                      tags={'nopad'})
        self._vect0_rhs = be.mpi_matrix_for_view(self._vect0_lhs)

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
        # Multiply beta by our sign to account for (potential) interchange
        beta = self._cfg.getfloat('constants', 'beta')*self._sign

        return self._be.kernel('conu_mpi', self.nvars,
                               self._scal0_lhs, self._scal0_rhs,
                               self._scal1_lhs, beta)

    def get_rsolve_kern(self):
        # Flux function constants
        gamma = self._cfg.getfloat('constants', 'gamma')
        mu = self._cfg.getfloat('constants', 'mu')
        pr = self._cfg.getfloat('constants', 'Pr')

        # Riemann solver constants
        beta = self._cfg.getfloat('constants', 'beta')*self._sign
        tau = self._cfg.getfloat('constants', 'tau')

        return self._be.kernel('rsolve_ldg_vis_mpi', self.ndims, self.nvars,
                               self._scal0_lhs, self._vect0_lhs,
                               self._scal0_rhs, self._vect0_rhs,
                               self._mag_pnorm_lhs, self._norm_pnorm_lhs,
                               gamma, mu, pr, beta, tau)
