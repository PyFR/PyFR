# -*- coding: utf-8 -*-

import numpy as np

def get_view_mats(interside, elemap):
    ninters = len(interside)

    scal = []
    for i in xrange(ninters):
        type, eidx, face, rtag = interside[i]
        ele = elemap[type]

        # After the += the length is increased by *three*
        scal += ele.get_scal_fpts_for_inter(eidx, face, rtag)

    # Concat the various numpy arrays together to yield the three matrices
    # required in order to define a view
    scal_v = [np.concatenate(scal[i::3])[np.newaxis,...] for i in xrange(3)]

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
    def __init__(self, be, elemap, cfg):
        self._be = be
        self._cfg = cfg

        # Get the number of dimensions and variables
        self.ndims = next(iter(elemap.viewvalues())).ndims
        self.nvars = next(iter(elemap.viewvalues())).nvars


class InternalInterfaces(BaseInterfaces):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super(InternalInterfaces, self).__init__(be, elemap, cfg)

        # Generate the left and right hand side view matrices
        scal_lhs = get_view_mats(lhs, elemap)
        scal_rhs = get_view_mats(rhs, elemap)

        # Allocate these on the backend as views
        self._scal_lhs = be.view(*scal_lhs, vlen=self.nvars, tags={'nopad'})
        self._scal_rhs = be.view(*scal_rhs, vlen=self.nvars, tags={'nopad'})

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


    def get_rsolve_kern(self):
        gamma = self._cfg.getfloat('constants', 'gamma')

        return self._be.kernel('rsolve_rus_inv_int', self.ndims, self.nvars,
                               self._scal_lhs, self._scal_rhs,
                               self._mag_pnorm_lhs, self._mag_pnorm_rhs,
                               self._norm_pnorm_lhs, gamma)


class MPIInterfaces(BaseInterfaces):
    # Tag used for MPI
    MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super(MPIInterfaces, self).__init__(be, elemap, cfg)
        self._rhsrank = rhsrank

        # Generate the left hand view matrices
        scal_lhs = get_view_mats(lhs, elemap)

        # Compute the total amount of data we will be 'viewing'
        nmpicol = scal_lhs[0].shape[1] * self.nvars

        # Allocate on the backend
        self._scal_lhs = be.mpi_view(*scal_lhs, vlen=self.nvars, tags={'nopad'})
        self._scal_rhs = be.mpi_matrix((1, nmpicol))

        # Get the left hand side physical normal data
        mag_pnorm_lhs = get_mag_pnorm_mat(lhs, elemap)
        norm_pnorm_lhs = get_norm_pnorm_mat(lhs, elemap)

        # Allocate
        self._mag_pnorm_lhs = be.const_matrix(mag_pnorm_lhs, tags={'nopad'})
        self._norm_pnorm_lhs = be.const_matrix(norm_pnorm_lhs, tags={'nopad'})

    def get_rsolve_kern(self):
        gamma = float(self._cfg.get('constants', 'gamma'))

        return self._be.kernel('rsolve_rus_inv_mpi', self.ndims, self.nvars,
                               self._scal_lhs, self._scal_rhs,
                               self._mag_pnorm_lhs, self._norm_pnorm_lhs,
                               gamma)

    def get_pack_kern(self):
        return self._be.kernel('pack', self._scal_lhs)

    def get_send_pack_kern(self):
        return self._be.kernel('send_pack', self._scal_lhs,
                               self._rhsrank, self.MPI_TAG)

    def get_recv_pack_kern(self):
        return self._be.kernel('recv_pack', self._scal_rhs,
                               self._rhsrank, self.MPI_TAG)

    def get_unpack_kern(self):
        return self._be.kernel('unpack', self._scal_rhs)
