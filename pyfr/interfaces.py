# -*- coding: utf-8 -*-

import numpy as np

def gen_view_mats(interside, elemap):
    ninters = len(interside)

    scal, pnorm = [], []
    for i in xrange(ninters):
        type, eidx, face, rtag = interside[i]
        ele = elemap[type]

        # After the += the length is increased by *three*
        scal += ele.get_scal_fpts_for_inter(eidx, face, rtag)
        pnorm += ele.get_pnorm_fpts_for_inter(eidx, face, rtag)

    # Concat the various numpy arrays together to yield the three matrices
    # required in order to define a view
    scal_v = [np.concatenate(scal[i::3])[np.newaxis,...] for i in xrange(3)]
    pnorm_v = [np.concatenate(pnorm[i::3])[np.newaxis,...] for i in xrange(3)]

    return scal_v, pnorm_v

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
        scal_lhs, pnorm_lhs = gen_view_mats(lhs, elemap)
        scal_rhs, pnorm_rhs = gen_view_mats(rhs, elemap)

        # Allocate these on the backend
        self._scal_lhs = be.view(*scal_lhs, vlen=self.nvars, tags={'nopad'})
        self._scal_rhs = be.view(*scal_rhs, vlen=self.nvars, tags={'nopad'})
        self._pnorm_lhs = be.view(*pnorm_lhs, vlen=self.ndims, tags={'nopad'})
        self._pnorm_rhs = be.view(*pnorm_rhs, vlen=self.ndims, tags={'nopad'})

    def get_rsolve_kern(self):
        gamma = self._cfg.getfloat('constants', 'gamma')

        scal_lhs, scal_rhs = self._scal_lhs, self._scal_rhs
        pnorm_lhs, pnorm_rhs = self._pnorm_lhs, self._pnorm_rhs

        return self._be.kernel('rsolve_rus_inv_int', self.ndims, self.nvars,
                               scal_lhs, pnorm_lhs, scal_rhs, pnorm_rhs, gamma)


class MPIInterfaces(BaseInterfaces):
    # Tag used for MPI
    MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super(MPIInterfaces, self).__init__(be, elemap, cfg)
        self._rhsrank = rhsrank

        # Generate the left hand view matrices
        scal_lhs, pnorm_lhs = gen_view_mats(lhs, elemap)

        # Compute the total amount of data we will be 'viewing'
        nmpicol = scal_lhs[0].shape[1] * self.nvars

        # Allocate on the backend
        self._scal_lhs = be.mpi_view(*scal_lhs, vlen=self.nvars, tags={'nopad'})
        self._scal_rhs = be.mpi_matrix((1, nmpicol))
        self._pnorm_lhs = be.view(*pnorm_lhs, vlen=self.ndims, tags={'nopad'})

    def get_rsolve_kern(self):
        gamma = float(self._cfg.get('constants', 'gamma'))

        scal_lhs, scal_rhs = self._scal_lhs, self._scal_rhs
        pnorm_lhs = self._pnorm_lhs

        return self._be.kernel('rsolve_rus_inv_mpi', self.ndims, self.nvars,
                               scal_lhs, scal_rhs, pnorm_lhs, gamma)

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
