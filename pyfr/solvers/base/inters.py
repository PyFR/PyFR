import numpy as np

from pyfr.util import first


def _get_inter_arrays(interside, meth, elemap, perm=Ellipsis):
    parts, reorder = [], []

    for etype, fidx, eidxs, idx in interside.foreach():
        parts.append(getattr(elemap[etype], meth)(eidxs, fidx))
        reorder.append(np.repeat(idx, elemap[etype].nfacefpts[fidx]))

    if not parts:
        return []

    ro = np.argsort(np.concatenate(reorder), kind='stable')[perm]
    return [np.concatenate(a)[ro] for a in zip(*parts)]


class BaseInters:
    def __init__(self, be, lhs, elemap, cfg):
        self._be = be
        self.elemap = elemap
        self.cfg = cfg

        # Get the number of dimensions and variables
        self.ndims = first(elemap.values()).ndims
        self.nvars = first(elemap.values()).nvars

        # Get the number of interfaces and flux points
        self.ninters = len(lhs)
        self.ninterfpts = sum(elemap[et].nfacefpts[fi]*len(ei)
                              for et, fi, ei in lhs.items())

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

    def set_external(self, name, spec, value=None):
        self._external_args[name] = spec

        if value is not None:
            self._external_vals[name] = value

    def _const_mat(self, inter, meth):
        m = _get_inter_arrays(inter, meth, self.elemap, self._perm)
        if not m:
            m = np.empty((0, self.ndims))
        else:
            m = m[0]

        return self._be.const_matrix(np.atleast_2d(m.T))

    def _get_perm_for_view(self, inter, meth):
        vm = _get_inter_arrays(inter, meth, self.elemap)
        mm = self._be.view(*vm, vshape=()).mapping.get()

        return np.argsort(mm[0])

    def _get_perm_for_field(self, inter, field):
        matmap, rowmap, colmap, reorder = [], [], [], []

        for etype, fidx, eidxs, idx in inter.foreach():
            mat = field[etype]
            eles = self.elemap[etype]
            fpts = eles.srtd_face_fpts[fidx][eidxs]
            nfp = fpts.shape[1]
            n = len(eidxs)

            matmap.append(np.full(n * nfp, mat.mid))
            rowmap.append(fpts.ravel())
            colmap.append(np.repeat(eidxs, nfp))
            reorder.append(np.repeat(idx, nfp))

        ro = np.argsort(np.concatenate(reorder), kind='stable')
        m = np.concatenate(matmap)[ro]
        r = np.concatenate(rowmap)[ro]
        c = np.concatenate(colmap)[ro]
        mm = self._be.view(m, r, c, vshape=()).mapping.get()
        return np.argsort(mm[0])

    def _view(self, inter, meth, vshape=(), with_perm=True):
        perm = self._perm if with_perm else Ellipsis
        vm = _get_inter_arrays(inter, meth, self.elemap, perm)
        return self._be.view(*vm, vshape=vshape)

    def _scal_view(self, inter, meth):
        return self._view(inter, meth, (self.nvars,))

    def _vect_view(self, inter, meth):
        return self._view(inter, meth, (self.ndims, self.nvars))

    def _xchg_view(self, inter, meth, vshape=(), with_perm=True):
        perm = self._perm if with_perm else Ellipsis
        vm = _get_inter_arrays(inter, meth, self.elemap, perm)
        return self._be.xchg_view(*vm, vshape=vshape)

    def _scal_xchg_view(self, inter, meth):
        return self._xchg_view(inter, meth, (self.nvars,))

    def _vect_xchg_view(self, inter, meth):
        return self._xchg_view(inter, meth, (self.ndims, self.nvars))

    def setup(self, sdata):
        pass

    @classmethod
    def serialisefn(cls, iface, prefix, srl):
        pass
