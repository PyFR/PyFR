# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


def get_view_mats(interside, mat, elemap, perm=Ellipsis):
    # Map from element type to view mat getter
    viewmatmap = {type: getattr(ele, mat) for type, ele in elemap.items()}

    scal = []
    for type, eidx, face, flags in interside:
        # After the += the length is increased by *three*
        scal += viewmatmap[type](eidx, face)

    # Concat the various numpy arrays together to yield the three matrices
    # required in order to define a view
    scal_v = [np.concatenate(scal[i::3], axis=1) for i in xrange(3)]

    # Permute
    scal_v = [sv[:,perm] for sv in scal_v]

    return scal_v


def get_mat(interside, mat, elemap, perm=Ellipsis):
    # Map from element type to view mat getter
    emap = {type: getattr(ele, mat) for type, ele in elemap.items()}

    # Get the matrix, swizzle the dimensions, and permute
    m = [emap[type](eidx, fidx) for type, eidx, fidx, flags in interside]
    m = np.concatenate(m)
    m = np.atleast_2d(m.T)
    m = m[:,perm]

    return m


def get_opt_view_perm(interside, mat, elemap):
    matmap, rcmap, stridemap = get_view_mats(interside, mat, elemap)

    # Since np.lexsort can not currently handle np.object arrays we
    # work around this by using id() to map each distinct matrix
    # object to an integer
    uid = np.vectorize(id)(matmap)

    # Sort
    return np.lexsort((uid[0], rcmap[0,:,1], rcmap[0,:,0]))


class BaseInters(object):
    __metaclass__ = ABCMeta

    def __init__(self, be, lhs, elemap, cfg):
        self._be = be
        self._elemap = elemap
        self._cfg = cfg

        # Get the number of dimensions and variables
        self.ndims = next(iter(elemap.viewvalues())).ndims
        self.nvars = next(iter(elemap.viewvalues())).nvars

        # Get the number of interfaces
        self.ninters = len(lhs)

        # Compute the total number of interface flux points
        self.ninterfpts = sum(elemap[etype].nfacefpts[fidx]
                              for etype, eidx, fidx, flags in lhs)

        # By default do not permute any of the interface arrays
        self._perm = Ellipsis

        # Kernel constants
        self._tpl_c = cfg.items_as('constants', float)

    def _const_mat(self, inter, meth):
        m = get_mat(inter, meth, self._elemap, self._perm)
        return self._be.const_matrix(m)

    def _view_onto(self, inter, meth):
        vm = get_view_mats(inter, meth, self._elemap, self._perm)
        return self._be.view(*vm, vlen=self.nvars)

    def _mpi_view_onto(self, inter, meth):
        vm = get_view_mats(inter, meth, self._elemap)
        return self._be.mpi_view(*vm, vlen=self.nvars)

    @abstractmethod
    def get_comm_flux_kern(self):
        pass
