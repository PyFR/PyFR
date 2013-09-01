# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


def get_view_mats(interside, mat, elemap, perm=Ellipsis):
    # Map from element type to view mat getter
    viewmatmap = {type: getattr(ele, mat) for type, ele in elemap.items()}

    scal = []
    for type, eidx, face, rtag in interside:
        # After the += the length is increased by *three*
        scal += viewmatmap[type](eidx, face, rtag)

    # Concat the various numpy arrays together to yield the three matrices
    # required in order to define a view
    scal_v = [np.hstack(scal[i::3]) for i in xrange(3)]

    # Permute
    scal_v = [sv[:,perm] for sv in scal_v]

    return scal_v


def get_mat(interside, mat, elemap, perm=Ellipsis):
    # Map from element type to view mat getter
    emap = {type: getattr(ele, mat) for type, ele in elemap.items()}

    # Form the matrix and permute
    m = [emap[type](eidx, fidx, rtag) for type, eidx, fidx, rtag in interside]
    m = np.concatenate(m)[None,...]
    m = m[:,perm,...]

    return m


def get_opt_view_perm(interside, mat, elemap):
    matmap, rcmap, stridemap = get_view_mats(interside, mat, elemap)

    # Since np.lexsort can not currently handle np.object arrays we
    # work around this by using np.unique to build an array in which
    # each distinct matrix object is represented by an integer
    u, uix = np.unique(matmap, return_inverse=True)

    # Sort
    return np.lexsort((uix, rcmap[0,:,1], rcmap[0,:,0]))


class BaseInters(object):
    __metaclass__ = ABCMeta

    def __init__(self, be, elemap, cfg):
        self._be = be
        self._elemap = elemap
        self._cfg = cfg

        # Get the number of dimensions and variables
        self.ndims = next(iter(elemap.viewvalues())).ndims
        self.nvars = next(iter(elemap.viewvalues())).nvars

        # By default do not permute any of the interface arrays
        self._perm = Ellipsis

    def _const_mat(self, inter, meth):
        m = get_mat(inter, meth, self._elemap, self._perm)
        return self._be.const_matrix(m)

    def _view_onto(self, inter, meth):
        vm = get_view_mats(inter, meth, self._elemap, self._perm)
        return self._be.view(*vm, vlen=self.nvars)

    def _mpi_view_onto(self, inter, meth):
        vm = get_view_mats(inter, meth, self._elemap)
        return self._be.mpi_view(*vm, vlen=self.nvars)

    def _kernel_constants(self):
        return self._cfg.items_as('constants', float)

    @abstractmethod
    def get_comm_flux_kern(self):
        pass
