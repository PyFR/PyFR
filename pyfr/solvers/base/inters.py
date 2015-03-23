# -*- coding: utf-8 -*-

import numpy as np


def _get_inter_objs(interside, getter, elemap):
    # Map from element type to view mat getter
    emap = {type: getattr(ele, getter) for type, ele in elemap.items()}

    # Get the data from the interface
    return [emap[type](eidx, fidx) for type, eidx, fidx, flags in interside]


def get_opt_view_perm(interside, mat, elemap):
    vm = _get_inter_objs(interside, mat, elemap)
    matmap, rcmap = [np.concatenate([m[i] for m in vm]) for i in range(2)]

    # Sort
    return np.lexsort((matmap, rcmap[:,1], rcmap[:,0]))


class BaseInters(object):
    def __init__(self, be, lhs, elemap, cfg):
        self._be = be
        self._elemap = elemap
        self.cfg = cfg

        # Get the number of dimensions and variables
        self.ndims = next(iter(elemap.values())).ndims
        self.nvars = next(iter(elemap.values())).nvars

        # Get the number of interfaces
        self.ninters = len(lhs)

        # Compute the total number of interface flux points
        self.ninterfpts = sum(elemap[etype].nfacefpts[fidx]
                              for etype, eidx, fidx, flags in lhs)

        # By default do not permute any of the interface arrays
        self._perm = Ellipsis

        # Kernel constants
        self._tpl_c = cfg.items_as('constants', float)

        # Kernels we provide
        self.kernels = {}

    def _const_mat(self, inter, meth):
        m = _get_inter_objs(inter, meth, self._elemap)

        # Swizzle the dimensions and permute
        m = np.concatenate(m)
        m = np.atleast_2d(m.T)
        m = m[:,self._perm]

        return self._be.const_matrix(m)

    def _view(self, inter, meth, vshape):
        vm = _get_inter_objs(inter, meth, self._elemap)
        vm = [np.concatenate(m)[self._perm] for m in zip(*vm)]
        return self._be.view(*vm, vshape=vshape)

    def _scal_view(self, inter, meth):
        return self._view(inter, meth, (self.nvars,))

    def _vect_view(self, inter, meth):
        return self._view(inter, meth, (self.ndims, self.nvars))

    def _xchg_view(self, inter, meth, vshape):
        vm = _get_inter_objs(inter, meth, self._elemap)
        vm = [np.concatenate(m)[self._perm] for m in zip(*vm)]
        return self._be.xchg_view(*vm, vshape=vshape)

    def _scal_xchg_view(self, inter, meth):
        return self._xchg_view(inter, meth, (self.nvars,))

    def _vect_xchg_view(self, inter, meth):
        return self._xchg_view(inter, meth, (self.ndims, self.nvars))
