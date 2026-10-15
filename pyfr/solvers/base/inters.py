import numpy as np

from pyfr.util import first


def sort_inter_pts(idxs, nfps, parts, perm=Ellipsis):
    if not parts:
        return []

    # Number of points of each interface and its offset in interface order
    cnt = np.zeros(sum(len(idx) for idx in idxs), dtype=int)
    for idx, nfp in zip(idxs, nfps):
        cnt[idx] = nfp
    offs = np.cumsum(cnt) - cnt

    # Destination of each point of each group of interfaces
    dests = [(offs[idx, None] + np.arange(nfp)).ravel()
             for idx, nfp in zip(idxs, nfps)]
    dest = np.concatenate(dests)

    # Invert the destinations to obtain the source of each point and permute
    ro = np.empty_like(dest)
    ro[dest] = np.arange(len(dest))
    ro = ro[perm]

    return [np.concatenate(a)[ro] for a in zip(*parts)]


def _get_inter_arrays(interside, meth, elemap, perm=Ellipsis):
    idxs, nfps, parts = [], [], []

    for etype, fidx, eidxs, idx in interside.foreach():
        idxs.append(idx)
        nfps.append(elemap[etype].nfacefpts[fidx])
        parts.append(getattr(elemap[etype], meth)(eidxs, fidx))

    return sort_inter_pts(idxs, nfps, parts, perm)


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

    def side_perm(self, inter, with_perm=True):
        return self._perm if with_perm else Ellipsis

    def _inter_ploc(self, inter):
        ploc, = _get_inter_arrays(inter, 'get_ploc_for_inters', self.elemap)
        return ploc

    def _const_mat(self, inter, meth):
        m = _get_inter_arrays(inter, meth, self.elemap, self.side_perm(inter))
        if not m:
            m = np.empty((0, self.ndims))
        else:
            m = m[0]

        return self._be.const_matrix(np.atleast_2d(m.T))

    def _get_perm_for_view(self, inter, meth):
        vm = _get_inter_arrays(inter, meth, self.elemap)
        mm = self._be.view(*vm, vshape=()).mapping.get()

        return np.argsort(mm[0][self.side_perm(inter, False)])

    def _get_perm_for_field(self, inter, field, blksz=None):
        idxs, nfps, parts = [], [], []

        for etype, fidx, eidxs, idx in inter.foreach():
            mat = field[etype]
            eles = self.elemap[etype]
            fpts = eles.srtd_face_fpts[fidx][eidxs]
            nfp = fpts.shape[1]
            n = len(eidxs)

            idxs.append(idx)
            nfps.append(nfp)
            parts.append((np.full(n*nfp, mat.mid), fpts.ravel(),
                          np.repeat(eidxs, nfp)))

        m, r, c = sort_inter_pts(idxs, nfps, parts)
        mm = self._be.view(m, r, c, vshape=()).mapping.get()
        sp = self.side_perm(inter, False)

        # Group the points by element block so partner data stays cached
        if blksz:
            return np.lexsort((mm[0][sp], c[sp] // blksz))
        # Otherwise order the points by their address
        else:
            return np.argsort(mm[0][sp])

    def _view(self, inter, meth, vshape=(), with_perm=True):
        perm = self.side_perm(inter, with_perm)
        vm = _get_inter_arrays(inter, meth, self.elemap, perm)
        return self._be.view(*vm, vshape=vshape)

    def _scal_view(self, inter, meth):
        return self._view(inter, meth, (self.nvars,))

    def _vect_view(self, inter, meth):
        return self._view(inter, meth, (self.ndims, self.nvars))

    def _xchg_view(self, inter, meth, vshape=(), with_perm=True):
        perm = self.side_perm(inter, with_perm)
        vm = _get_inter_arrays(inter, meth, self.elemap, perm)
        return self._be.xchg_view(*vm, vshape=vshape)

    def _scal_xchg_view(self, inter, meth):
        return self._xchg_view(inter, meth, (self.nvars,))

    def _vect_xchg_view(self, inter, meth):
        return self._xchg_view(inter, meth, (self.ndims, self.nvars))

    def setup(self, sdata, prevcfg):
        pass

    @classmethod
    def serialisefn(cls, iface, prefix, srl):
        pass
