from collections import namedtuple

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import DistributedDirectory, get_comm_rank_root, mpi
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape, LineShape, QuadShape, TriShape, proj_pts
from pyfr.util import subclass_where


class SubDOFMap:
    # Quad-face corner positions in the local (i, j) frame
    _qpos = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    # Per-kind packing spec: (nverts, npos)
    _kind_specs = {'corner': (1, 0), 'edge': (2, 1), 'qface': (4, 2),
                   'tface': (3, 2)}

    def __init__(self, n, npts, items, body):
        self.n = n
        self.npts = npts
        self.body_idxs = np.array(body, dtype=int)

        # Pack only kinds with entities; absent kinds simply not in the dict
        self._kinds = {k: self._pack(its, *self._kind_specs[k])
                       for k, its in items.items() if its}

    @property
    def kinds(self):
        return self._kinds.keys()

    @staticmethod
    def _pack(items, nverts, npos):
        vdofs = sorted(items)
        arr = np.empty((len(vdofs), nverts + npos), dtype=int)
        for i, s in enumerate(vdofs):
            arr[i] = items[s]

        return np.array(vdofs, dtype=int), arr[:, :nverts], arr[:, nverts:]

    def canonical_vdofs(self, kind, cnodes):
        # Per-(ele, vdof) canonical content; same-entity vdofs share keys
        return getattr(self, f'_{kind}_canonical')(cnodes)

    def _fsrc(self, vdofs, neles):
        eoff = (np.arange(neles)*self.npts)[:, None]
        return (eoff + vdofs[None, :]).ravel()

    def _corner_canonical(self, cn):
        vdofs, lin, _ = self._kinds['corner']
        neles = len(cn)

        # Each corner is its own entity; key is the cnode itself
        keys = cn[:, lin[:, 0]].reshape(-1, 1)
        cpos = np.zeros(neles*len(vdofs), dtype=int)
        return keys, self._fsrc(vdofs, neles), cpos, 1

    def _edge_canonical(self, cn):
        vdofs, ec, pos = self._kinds['edge']
        neles = len(cn)
        nint = self.n - 1

        # Per-vdof edge cnode pair; sort and flip k accordingly
        c0 = cn[:, ec[:, 0]].ravel()
        c1 = cn[:, ec[:, 1]].ravel()
        k = np.tile(pos.ravel(), neles)
        sw = c0 > c1

        keys = np.column_stack([np.where(sw, c1, c0), np.where(sw, c0, c1)])
        cpos = np.where(sw, self.n - k, k) - 1
        return keys, self._fsrc(vdofs, neles), cpos, nint

    def _qface_canonical(self, cn):
        vdofs, fcn, pos = self._kinds['qface']
        neles = len(cn)
        npa = self.n - 1
        nint = npa*npa

        # Dedup fcn so orientation runs once per (ele, qface) not per vdof
        fcn_q, qidx = np.unique(fcn, axis=0, return_inverse=True)

        # Lowest-corner origin, lower-cnode neighbour as +i
        corners_q = cn[:, fcn_q]
        o = np.argmin(corners_q, axis=-1)
        na, nb = o ^ 1, o ^ 2
        ca = np.take_along_axis(corners_q, na[..., None], -1).squeeze(-1)
        cb = np.take_along_axis(corners_q, nb[..., None], -1).squeeze(-1)
        sw = ca > cb
        a, b = np.where(sw, nb, na), np.where(sw, na, nb)

        qpos = self._qpos*self.n
        po, vi, vj = qpos[o], qpos[a] - qpos[o], qpos[b] - qpos[o]

        # Project per-vdof (i, j) onto each vdof's qface canonical frame
        # via fused dot-products (no intermediate allocation)
        d = pos[None] - po[:, qidx]
        ci = np.einsum('...j,...j->...', d, vi[:, qidx]) // self.n
        cj = np.einsum('...j,...j->...', d, vj[:, qidx]) // self.n

        keys = np.sort(corners_q, axis=-1)[:, qidx].reshape(-1, 4)
        cpos = ((ci - 1)*npa + (cj - 1)).ravel()
        return keys, self._fsrc(vdofs, neles), cpos, nint

    def _tface_canonical(self, cn):
        vdofs, fcn, bc = self._kinds['tface']
        neles = len(cn)
        npa = self.n - 1
        nint = npa*(npa - 1) // 2

        # Dedup fcn so argsort runs once per (ele, tface) not per vdof
        fcn_t, tidx = np.unique(fcn, axis=0, return_inverse=True)
        corners_t = cn[:, fcn_t]
        perm_t = np.argsort(corners_t, axis=2)
        keys_t = np.take_along_axis(corners_t, perm_t, axis=2)

        # Per-vdof barycentric, permuted by each vdof's tface sort
        barys = np.tile(bc, (neles, 1)).reshape(neles, len(vdofs), 2)
        bary = np.concatenate([self.n - barys.sum(axis=2, keepdims=True),
                               barys], axis=2)
        sbary = np.take_along_axis(bary, perm_t[:, tidx], axis=2)

        # Triangular flatten of (sb, sc) into [0, nint)
        sb, sc = sbary[..., 1], sbary[..., 2]
        cpos = ((sb - 1)*npa - sb*(sb - 1) // 2 + sc - 1).ravel()
        keys = keys_t[:, tidx].reshape(-1, 3)
        return keys, self._fsrc(vdofs, neles), cpos, nint


class BaseSubdivShape:
    _face_shapes = {'line': LineShape, 'quad': QuadShape, 'tri': TriShape}

    def __init__(self, n):
        self.n = n

    @staticmethod
    def _int_lattice(pts, n):
        return np.rint((pts + 1)*n/2).astype(int)

    @memoize
    def _kind_info(self, kind):
        fshape = self._face_shapes[kind]
        fhpts = fshape.std_ele(self.n)
        flpts = fshape.std_ele(1)

        # Identify linear vertices on the face
        basis = get_polybasis(kind, 1, flpts)
        op = basis.nodal_basis_at(fhpts)
        supp = [tuple(np.flatnonzero(r)) for r in np.abs(op) > 1e-12]

        return fhpts, flpts, op, supp, self._int_lattice(fhpts, self.n)

    def topology(self, svpts):
        # Sub-DOF entity classification (corner / edge / qface / tface / body)
        # for the subdivision of an element of this shape at the given svpts.
        # Used by CleanToGrid to dedup coincident sub-points; no VTK in here.
        n = self.n
        shapecls = subclass_where(BaseShape, name=self.name)
        cornpos = self._int_lattice(shapecls.std_ele(1), 2*n)

        keys = self._int_lattice(svpts, 2*n).tolist()
        vdofmap = {tuple(k): i for i, k in enumerate(keys)}

        items = {'corner': {}, 'edge': {}, 'qface': {}, 'tface': {}}
        touched = set()

        # Classify each face DOF by its linear support footprint
        for kind, proj, _ in shapecls.faces:
            fpts, flins, op, supp, pos = self._kind_info(kind)
            pkeys = self._int_lattice(proj_pts(proj, fpts), 2*n).tolist()
            fcornpos = self._int_lattice(proj_pts(proj, flins), 2*n)
            fverts = (fcornpos[:, None] == cornpos).all(-1).argmax(1)

            for pkey, fs, row, fp in zip(pkeys, supp, op, pos):
                vdof = vdofmap[tuple(pkey)]
                touched.add(vdof)
                verts = tuple(fverts[j] for j in fs)

                if len(fs) == 1:
                    items['corner'][vdof] = verts
                elif len(fs) == 2:
                    items['edge'][vdof] = (*verts, int(row[fs[1]]*n + 0.25))
                elif len(fs) == 3:
                    items['tface'][vdof] = (*verts, *fp)
                else:
                    items['qface'][vdof] = (*verts, *fp)

        body = sorted(set(range(len(svpts))) - touched)
        return SubDOFMap(self.n, len(svpts), items, body)


class QuadSubdivShape(BaseSubdivShape):
    name = 'quad'
    ndims = 2


class HexSubdivShape(BaseSubdivShape):
    name = 'hex'
    ndims = 3


class TriSubdivShape(BaseSubdivShape):
    name = 'tri'
    ndims = 2


class TetSubdivShape(BaseSubdivShape):
    name = 'tet'
    ndims = 3


class PriSubdivShape(BaseSubdivShape):
    name = 'pri'
    ndims = 3


class PyrSubdivShape(BaseSubdivShape):
    name = 'pyr'
    ndims = 3


def get_subdiv_shape(name, n):
    return subclass_where(BaseSubdivShape, name=name)(n)


KindGlobal = namedtuple('KindGlobal',
                        'keys defidx nint emap count reducer')


def _unique_rows(keys):
    n, k = keys.shape
    if k == 1:
        uniq, inv = np.unique(keys.ravel(), return_inverse=True)
        return uniq.reshape(-1, 1), inv

    # lexsort + first-of-group is faster than struct-view for k>1
    order = np.lexsort(keys.T)
    srt = keys[order]
    fmask = np.empty(n, dtype=bool)
    fmask[:1] = True
    fmask[1:] = (srt[1:] != srt[:-1]).any(axis=1)
    inv = np.empty(n, dtype=int)
    inv[order] = np.cumsum(fmask) - 1
    return srt[fmask], inv


class Reducer:
    # FxHash multiplicative-XOR constants (golden ratio and sign-bit mask)
    FX_SEED = np.uint64(0x9E3779B97F4A7C15)
    FX_MASK = np.uint64(0x7FFFFFFFFFFFFFFF)

    def __init__(self, comm, fkeys):
        self.dd = DistributedDirectory(comm, self._hash_rows(fkeys))
        self.uniq, self.inv = _unique_rows(self.dd.scatter(fkeys))

    def __call__(self, values):
        rv = self.dd.scatter(values)
        gv = np.zeros((len(self.uniq), *values.shape[1:]), dtype=values.dtype)
        np.add.at(gv, self.inv, rv)
        return self.dd.gather(gv[self.inv])

    @classmethod
    def _hash_rows(cls, keys):
        h = np.zeros(len(keys), dtype=np.uint64)
        for col in keys.view(np.uint64).T:
            h = (h*cls.FX_SEED) ^ col

        return h & cls.FX_MASK


class CleanToGrid:
    @staticmethod
    def _class_rows(classes, nint):
        return (classes[:, None]*nint + np.arange(nint)).ravel()

    def __init__(self, cnodemap, divmap, nsvptsmap, shared):
        self.comm, _, _ = get_comm_rank_root()
        self.cnodemap = cnodemap

        topos, edef, ktypes = {}, {}, set()
        for etype, cnodes in cnodemap.items():
            shape = get_subdiv_shape(etype, divmap[etype])
            topos[etype] = shape.topology(nsvptsmap[etype])
            edef[etype] = np.isin(cnodes, shared).any(axis=1)
            ktypes.update(topos[etype].kinds)

        # Reduce to obtain the global set of entities
        ktypes = set().union(*self.comm.allgather(ktypes))

        # Per-kind cross-etype dedup -> self.kinds (with emap populated)
        self.kinds = {kind: self._init_kind(kind, topos, cnodemap, edef)
                      for kind in sorted(ktypes)}

        self.layouts = {etype: self._build_layout(etype, topos[etype])
                        for etype in cnodemap}

    def _init_kind(self, kind, topos, cnodemap, edef):
        evdofs, klist, dlist, nint, ncols = {}, [], [], 0, 0
        for etype, cn in cnodemap.items():
            if kind not in topos[etype].kinds:
                continue

            keys, fsrc, cpos, nint = topos[etype].canonical_vdofs(kind, cn)
            evdofs[etype] = (keys, fsrc, cpos)
            klist.append(keys)
            dlist.append(np.repeat(edef[etype], len(keys) // len(cn)))
            ncols = keys.shape[1]

        # Agree on nint and key width across ranks
        nint = self.comm.allreduce(nint, op=mpi.MAX)
        ncols = self.comm.allreduce(ncols, op=mpi.MAX)

        # Deduplicate canonical keys across all contributing etypes
        akeys = np.concatenate(klist or [np.empty((0, ncols), dtype=int)])
        adefs = np.concatenate(dlist or [np.empty(0, dtype=bool)])

        uniq, inv = _unique_rows(akeys)
        defidx = np.unique(inv[adefs])

        # Map each etype's vdofs to their global class indices
        emap, off = {}, 0
        for etype, (keys, fsrc, cpos) in evdofs.items():
            cls = inv[off:off + len(keys)]
            emap[etype] = (np.unique(cls), fsrc, cls*nint + cpos)
            off += len(keys)

        # Determine counts
        count = np.bincount(inv, minlength=len(uniq)) // max(nint, 1)
        reducer = Reducer(self.comm, uniq[defidx])
        count[defidx] = reducer(count[defidx])

        return KindGlobal(uniq, defidx, nint, emap, count, reducer)

    def _build_layout(self, etype, topo):
        neles, npts = len(self.cnodemap[etype]), topo.npts
        remap = np.empty((neles, npts), dtype=int)
        kept, off = [], 0

        # Merge coincident vdofs into unique kept positions
        for g in self.kinds.values():
            if (kp := g.emap.get(etype)) is None:
                continue

            # Local vdof positions within this etype's classes
            classes, fsrc, fdst = kp
            cls, cpos = divmod(fdst, g.nint)
            lcls = np.searchsorted(classes, cls)
            within = lcls*g.nint + cpos

            # First-occurrence source indices for each kept vdof
            ukept, fidx = np.unique(within, return_index=True)
            ksrc = np.empty(len(classes)*g.nint, dtype=int)
            ksrc[ukept] = fsrc[fidx]

            # Update the remap table and accumulate kept sources
            remap[divmod(fsrc, npts)] = off + within
            kept.append(ksrc)
            off += len(classes)*g.nint

        # Body vdofs: unique by construction (sequential kept ids)
        bidxs = topo.body_idxs
        bpos = ((np.arange(neles)*npts)[:, None] + bidxs).ravel()
        seq = np.arange(bpos.size).reshape(neles, len(bidxs))
        remap[:, bidxs] = off + seq
        kept.append(bpos)

        return remap, np.concatenate(kept), bpos

    def select(self, etype, values):
        _, kept, _ = self.layouts[etype]
        flat = values.swapaxes(0, 1).reshape(-1, *values.shape[2:])
        return flat[kept]

    def average(self, fields, ncomp, dtype):
        # Per-kind: accumulate sums -> exchange deficient sums -> divide by
        # precomputed topological count
        vdofvals = {etype: vals.swapaxes(0, 1).reshape(-1, ncomp)
                    for etype, vals in fields.items()}

        out = {etype: [] for etype in fields}
        for g in self.kinds.values():
            psum = np.zeros((len(g.keys)*g.nint, ncomp), dtype=dtype)
            for etype, (_, fsrc, fdst) in g.emap.items():
                np.add.at(psum, fdst, vdofvals[etype][fsrc])

            rows = self._class_rows(g.defidx, g.nint)
            vals = psum[rows].reshape(-1, g.nint, ncomp)
            psum[rows] = g.reducer(vals).reshape(-1, ncomp)

            psum /= np.repeat(g.count, g.nint)[:, None]
            for etype, (classes, _, _) in g.emap.items():
                # When this etype's classes cover every global class for
                # this kind, the fancy-index slice is the whole array
                if len(classes) == len(g.keys):
                    out[etype].append(psum)
                else:
                    rows = self._class_rows(classes, g.nint)
                    out[etype].append(psum[rows])

        # Append body slice and concat per-etype
        for etype in fields:
            _, _, body = self.layouts[etype]
            out[etype].append(vdofvals[etype][body])
            out[etype] = np.concatenate(out[etype])

        return out

    def renormalize(self, arr):
        # Averaging at shared vertices de-normalizes; restore unit length.
        return arr / np.linalg.norm(arr, axis=0, keepdims=True)


class NullCleaner:
    layouts = None

    def average(self, stacked, ncomp, dtype):
        # Passthrough — preserve input dtype so raw-mode precision matches
        # the pre-PR behavior (no dedup, no cast).
        return {k: a.swapaxes(0, 1).reshape(-1, ncomp)
                for k, a in stacked.items()}

    def select(self, etype, arr):
        return arr.swapaxes(0, 1).reshape(-1, arr.shape[-1])

    def renormalize(self, arr):
        # No averaging happens, so per-element unit vectors stay unit.
        return arr
