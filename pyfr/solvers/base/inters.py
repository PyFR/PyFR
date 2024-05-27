import numpy as np

from pyfr.util import first


def _get_inter_objs(interside, getter, elemap):
    # Map from element type to view mat getter
    emap = {type: getattr(ele, getter) for type, ele in elemap.items()}

    # Get the data from the interface
    return [emap[type](eidx, fidx) for type, eidx, fidx in interside]


class BaseInters:
    def __init__(self, be, lhs, elemap, cfg):
        self._be = be
        self.elemap = elemap
        self.cfg = cfg

        # Get the number of dimensions and variables
        self.ndims = first(elemap.values()).ndims
        self.nvars = first(elemap.values()).nvars

        # Get the number of interfaces
        self.ninters = len(lhs)

        # Compute the total number of interface flux points
        self.ninterfpts = sum(elemap[etype].nfacefpts[fidx]
                              for etype, eidx, fidx in lhs)

        # By default do not permute any of the interface arrays
        self._perm = Ellipsis

        # Kernel constants
        self.c = cfg.items_as('constants', float)

        # Kernels and MPI requests we provide
        self.kernels = {}
        self.mpireqs = {}

        # Global kernel arguments
        self._external_args = {}
        self._external_vals = {}

    def prepare(self, t):
        pass

    def _set_external(self, name, spec, value=None):
        self._external_args[name] = spec

        if value is not None:
            self._external_vals[name] = value

    def _const_mat(self, inter, meth):
        m = _get_inter_objs(inter, meth, self.elemap)

        # Swizzle the dimensions and permute
        m = np.concatenate(m)
        m = np.atleast_2d(m.T)
        m = m[:, self._perm]

        return self._be.const_matrix(m)

    def _get_perm_for_view(self, inter, meth):
        vm = _get_inter_objs(inter, meth, self.elemap)
        vm = [np.concatenate(m) for m in zip(*vm)]
        mm = self._be.view(*vm, vshape=()).mapping.get()

        return np.argsort(mm[0])

    def _view(self, inter, meth, vshape=(), with_perm=True):
        vm = _get_inter_objs(inter, meth, self.elemap)
        perm = self._perm if with_perm else Ellipsis
        vm = [np.concatenate(m)[perm] for m in zip(*vm)]
        return self._be.view(*vm, vshape=vshape)

    def _scal_view(self, inter, meth):
        return self._view(inter, meth, (self.nvars,))

    def _vect_view(self, inter, meth):
        return self._view(inter, meth, (self.ndims, self.nvars))

    def _xchg_view(self, inter, meth, vshape=(), with_perm=True):
        vm = _get_inter_objs(inter, meth, self.elemap)
        perm = self._perm if with_perm else Ellipsis
        vm = [np.concatenate(m)[perm] for m in zip(*vm)]
        return self._be.xchg_view(*vm, vshape=vshape)

    def _scal_xchg_view(self, inter, meth):
        return self._xchg_view(inter, meth, (self.nvars,))

    def _vect_xchg_view(self, inter, meth):
        return self._xchg_view(inter, meth, (self.ndims, self.nvars))
