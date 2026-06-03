from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.points import PointLocator, PointSampler
from pyfr.shapes import BaseShape, proj_pts
from pyfr.snapshot.sample import SnapshotSample
from pyfr.subdiv import CleanToGrid
from pyfr.util import subclass_where


def interp_ops(op, arr, *, cast=False):
    if cast:
        op = op.astype(arr.dtype, copy=False)
    return (op @ arr.reshape(arr.shape[0], -1)).reshape(op.shape[0],
                                                        *arr.shape[1:])


def _cfg_dtype(cfg):
    return np.float32 if cfg.get('backend', 'precision', 'single') == 'single' \
        else np.float64


def _cfg_elementscls(cfg):
    from pyfr.solvers.base import BaseSystem
    return subclass_where(BaseSystem, name=cfg.get('solver', 'system'))\
        .elementscls


class BaseSnapshotRegion:
    cleaner = None

    def __init__(self, mesh, cfg):
        self.mesh = mesh
        self.config = cfg
        self.ndims = mesh.ndims
        self.dtype = _cfg_dtype(cfg)
        self._snap_form = None

    def sample(self, snap):
        return SnapshotSample(self, snap)

    def _ensure_form(self, snap, build=None):
        new = (tuple(snap.pris_names), bool(snap.has_grads))
        if self._snap_form is None:
            self._snap_form = new
            if build is not None:
                build()
        elif self._snap_form != new:
            old_p, old_g = self._snap_form
            new_p, new_g = new
            raise RuntimeError(
                f'{type(self).__name__} was configured for '
                f'(pris_names={list(old_p)}, has_grads={old_g}); '
                f'cannot reuse with '
                f'(pris_names={list(new_p)}, has_grads={new_g})')

    def _build_geometry(self):
        pass

    def _compute_sample(self, sample, snap):
        pass

    def npts(self, etype):
        p = self.ploc[etype]
        return p.shape[1] if p.ndim == 2 else p.shape[1]*p.shape[2]

    def points(self, etype):
        p = self.ploc[etype]
        if p.ndim == 2:
            return np.ascontiguousarray(p.T)
        return np.ascontiguousarray(
            p.transpose(2, 1, 0).reshape(-1, p.shape[0]))

    def connectivity(self, etype, sub_nodes):
        p = self.ploc[etype]
        if p.ndim == 3:
            nsvpts, neles = p.shape[1], p.shape[2]
            con = np.tile(sub_nodes, (neles, 1))
            con += (np.arange(neles)*nsvpts)[:, None]
            return con
        if self.cleaner is None:
            raise NotImplementedError(
                f'{type(self).__name__} has no per-element cell layout')
        return self.cleaner.layouts[etype][0][:, sub_nodes]

    def cell_curved(self, etype):
        return None

    def _finalize_sample(self, sample, snap, raw_pris, raw_grad_pris):
        nvars = len(snap.pris_names)
        if self.clean:
            sample.pris = self._clean_pris(raw_pris, nvars)
            if any(v is None for v in raw_grad_pris.values()):
                sample.grad_pris = raw_grad_pris
            else:
                sample.grad_pris = self._clean_grad_pris(raw_grad_pris, nvars)
        else:
            sample.pris = raw_pris
            sample.grad_pris = raw_grad_pris

    def _clean_pris(self, raw, nvars):
        stacked = {k: np.stack(prims_list, axis=-1)
                   for k, prims_list in raw.items()}
        avg = self.cleaner.average(stacked, nvars, self.dtype)
        return {k: list(avg[k].T) for k in raw}

    def _clean_grad_pris(self, raw, nvars):
        ndims = self.ndims
        flat = {}
        for k, grads_list in raw.items():
            stacked = np.stack(grads_list, axis=0)       # (nvars,ndims,...)
            flat[k] = stacked.reshape(nvars*ndims, *stacked.shape[2:]
                                      ).transpose(1, 2, 0)

        avg = self.cleaner.average(flat, nvars*ndims, self.dtype)
        return {k: [c for c in avg[k].T.reshape(nvars, ndims, -1)] for k in raw}


class VolumeSnapshotRegion(BaseSnapshotRegion):
    def __init__(self, mesh, cfg, spec, refpts_fn, *, divisor=None, clean=True):
        super().__init__(mesh, cfg)

        if isinstance(spec, dict):
            self._eidxs = {et: (None if e is None or isinstance(e, slice)
                                else np.asarray(e))
                           for et, e in spec.items()}
        else:
            self._eidxs = {et: None for et in spec}
        self.etypes = list(self._eidxs)

        self._refpts_fn = refpts_fn
        self._divisor = divisor
        self.clean = clean

        self._build_geometry()

    def _ele_spts(self, et):
        spts = self.mesh.spts[et]
        eidxs = self._eidxs[et]
        return spts if eidxs is None else spts[:, eidxs]

    def _ele_spts_nodes(self, et):
        spts_nodes = self.mesh.spts_nodes[et]
        eidxs = self._eidxs[et]
        return spts_nodes if eidxs is None else spts_nodes[eidxs]

    def _ele_data(self, snap, et):
        arr = snap.data[et]
        eidxs = self._eidxs[et]
        return arr if eidxs is None else arr[..., eidxs]

    def _ele_grad_data(self, snap, et):
        if snap.grad_data is None:
            return None
        g = snap.grad_data.get(et)
        if g is None:
            return None
        eidxs = self._eidxs[et]
        return g if eidxs is None else g[..., eidxs]

    def _build_geometry(self):
        self._ops = {}
        self._refpts_at = {}
        for et in self.etypes:
            shapecls = subclass_where(BaseShape, name=et)
            nspts = self.mesh.spts[et].shape[0]
            shape = shapecls(nspts, self.config)
            pts = self._refpts_fn(shapecls, shape)
            self._refpts_at[et] = pts
            self._ops[et] = (shape.sbasis.nodal_basis_at(pts),
                             shape.ubasis.nodal_basis_at(pts))

        self.cleaner = self._build_cleaner() if self.clean else None

        self.ploc = self._build_ploc()

    def _build_cleaner(self):
        mesh = self.mesh
        cnodemap, divmap, svptsmap = {}, {}, {}
        for et in self.etypes:
            shapecls = subclass_where(BaseShape, name=et)
            spts_nodes = self._ele_spts_nodes(et)
            cidxs = shapecls.corner_pts_idxs(spts_nodes.shape[1])
            cnodemap[et] = spts_nodes[:, cidxs]
            divmap[et] = self._divisor
            svptsmap[et] = self._refpts_at[et]

        shared = np.fromiter(mesh.shared_nodes.by_node, dtype=int)
        return CleanToGrid(cnodemap, divmap, svptsmap, shared)

    def _build_ploc(self):
        out = {}
        for et in self.etypes:
            mesh_op, _ = self._ops[et]
            xd = interp_ops(mesh_op, self._ele_spts(et))   # (nsvpts,neles,ndims)
            if self.clean:
                out[et] = np.ascontiguousarray(self.cleaner.select(et, xd).T)
            else:
                out[et] = np.ascontiguousarray(xd.transpose(2, 0, 1))
        return out

    def cell_curved(self, etype):
        curved = self.mesh.spts_curved[etype]
        eidxs = self._eidxs[etype]
        return curved if eidxs is None else curved[eidxs]

    def _compute_sample(self, sample, snap):
        cfg = self.config

        # Interpolate stored data to refpts: (nrefpts, nvars, neles).
        interp_data = {et: interp_ops(self._ops[et][1],
                                   self._ele_data(snap, et)).swapaxes(0, 1)
                       for et in self.etypes}

        raw_pris = {et: snap.to_pris(interp_data[et], cfg)
                    for et in self.etypes}

        raw_grad_pris = {}
        for et in self.etypes:
            g = self._ele_grad_data(snap, et)
            if g is None:
                raw_grad_pris[et] = None
                continue

            _, soln_op = self._ops[et]
            cg = np.einsum('sp,dpvn->dsvn', soln_op, g)
            raw_grad_pris[et] = snap.to_grad_pris(
                interp_data[et], cg.transpose(2, 0, 1, 3), cfg)

        self._finalize_sample(sample, snap, raw_pris, raw_grad_pris)

        # Aux pass-through
        for et in self.etypes:
            eidxs = self._eidxs[et]
            for name, arr in snap.aux(et).items():
                sample.field_arrays[et, name] = (arr if eidxs is None
                                                 else arr[eidxs])


class SurfaceSnapshotRegion(BaseSnapshotRegion):
    def __init__(self, mesh, cfg, bcname, divisor, refpts_fn=None, *,
                 clean=True):
        super().__init__(mesh, cfg)

        self._divisor = divisor
        self._refpts_fn = (refpts_fn or
                           (lambda sc, _sh=None: sc.std_ele(self._divisor)))
        self.clean = clean

        bcnames = [bcname] if isinstance(bcname, str) else list(bcname)

        # Merge the connectivity from each boundary into one map
        merged = {}
        for bn in bcnames:
            conn = self.mesh.bcon.get(bn.removeprefix('bc/'))
            if conn is None:
                continue

            for etype, fidx, eidxs in conn.items():
                key = (etype, fidx)
                eidxs = np.asarray(eidxs)
                if key in merged:
                    merged[key] = np.concatenate([merged[key], eidxs])
                else:
                    merged[key] = eidxs

        self._conn = [(et, f, idx) for (et, f), idx in merged.items()]

        self._build_geometry()

    @property
    def etypes(self):
        return list(dict.fromkeys(g[3] for g in self._groups))

    def _build_groups(self):
        cfg, mesh = self.config, self.mesh
        groups = []
        for etype, fidx, eidxs in self._conn:
            shapecls = subclass_where(BaseShape, name=etype)
            nspts = mesh.spts[etype].shape[0]
            itype, proj, _ = shapecls.faces[fidx]

            face_refpts = self._refpts_fn(subclass_where(BaseShape, name=itype))
            fpts = proj_pts(proj, face_refpts)

            shape = shapecls(nspts, cfg)
            mesh_op = shape.sbasis.nodal_basis_at(fpts)
            soln_op = shape.ubasis.nodal_basis_at(fpts)

            groups.append((etype, fidx, eidxs, itype, mesh_op, soln_op, fpts))

        return groups

    def _build_geometry(self):
        self._groups = self._build_groups()

        elementscls = _cfg_elementscls(self.config)
        self._eles = []
        for etype, _, eidxs, *_ in self._groups:
            shapecls = subclass_where(BaseShape, name=etype)
            spts = self.mesh.spts[etype][:, eidxs]
            self._eles.append(elementscls(shapecls, spts, self.config))

        self.cleaner = self._build_cleaner() if self.clean else None

        self.ploc = self._build_ploc()
        self.normals = self._build_normals()
        self.wall_dist = self._build_wall_dist()

    def _assemble(self, fn):
        # Map fn(gi, group) over face-groups, concatenating same-itype results
        acc = defaultdict(list)
        for gi, g in enumerate(self._groups):
            acc[g[3]].append(fn(gi, g))

        cat = lambda p: np.ascontiguousarray(np.concatenate(p, axis=-1))
        out = {}
        for it, parts in acc.items():
            if parts[0] is None:
                out[it] = None
            elif isinstance(parts[0], list):
                out[it] = [cat(c) for c in zip(*parts)]
            else:
                out[it] = cat(parts)

        return out

    def _build_cleaner(self):
        mesh = self.mesh

        face_pieces = defaultdict(list)
        for etype, fidx, eidxs, itype, *_ in self._groups:
            shapecls = subclass_where(BaseShape, name=etype)
            spts_nodes = mesh.spts_nodes[etype]
            cidxs = shapecls.face_corner_pts_idxs(fidx,
                                                  spts_nodes.shape[1])
            face_pieces[itype].append(spts_nodes[np.ix_(eidxs, cidxs)])

        cnodemap, divmap, svptsmap = {}, {}, {}
        for itype, pieces in face_pieces.items():
            cnodemap[itype] = np.concatenate(pieces)
            divmap[itype] = self._divisor
            svptsmap[itype] = self._refpts_fn(
                subclass_where(BaseShape, name=itype))

        shared = np.fromiter(mesh.shared_nodes.by_node, dtype=int)
        return CleanToGrid(cnodemap, divmap, svptsmap, shared)

    def _build_ploc(self):
        def fn(gi, g):
            etype, _, eidxs, _, mesh_op, _, _ = g
            xd = interp_ops(mesh_op, self.mesh.spts[etype][:, eidxs])
            return xd.transpose(2, 0, 1)

        raw = self._assemble(fn)
        if not self.clean:
            return raw

        return {it: np.ascontiguousarray(
                    self.cleaner.select(it, xd.transpose(1, 2, 0)).T)
                for it, xd in raw.items()}

    def _build_normals(self):
        def fn(gi, g):
            fidx, fpts = g[1], g[6]
            eles = self._eles[gi]
            refnorm = eles.basis.faces[fidx][2]
            pn = eles.pnorm_at(fpts, np.tile(refnorm, (len(fpts), 1)))
            pn = pn.transpose(2, 0, 1)
            return pn / np.linalg.norm(pn, axis=0)

        raw = self._assemble(fn)
        if not self.clean:
            return raw

        stacked = {it: arr.transpose(1, 2, 0) for it, arr in raw.items()}
        avg = self.cleaner.average(stacked, self.ndims, self.dtype)
        out = {}
        for it in raw:
            n = avg[it].T
            n = n / np.linalg.norm(n, axis=0, keepdims=True)
            out[it] = np.ascontiguousarray(n)
        return out

    def _build_wall_dist(self):
        raw = self._assemble(
            lambda gi, g: self._eles[gi].min_upt_face_dist_approx(g[1]))

        if not self.clean:
            return raw

        stacked = {}
        for it, wd_per_ele in raw.items():
            nfpts = len(self._refpts_fn(subclass_where(BaseShape, name=it)))
            wd_per_fpt = np.broadcast_to(wd_per_ele[None, :, None],
                                         (nfpts, len(wd_per_ele), 1)).copy()
            stacked[it] = wd_per_fpt

        avg = self.cleaner.average(stacked, 1, self.dtype)
        return {it: avg[it].squeeze(-1) for it in raw}

    def _compute_sample(self, sample, snap):
        cfg = self.config

        cons = []
        for etype, _, eidxs, _, _, soln_op, _ in self._groups:
            c = interp_ops(soln_op, snap.data[etype][:, :, eidxs])
            cons.append(c.swapaxes(0, 1))

        raw_pris = self._assemble(lambda gi, g: snap.to_pris(cons[gi], cfg))

        def gp(gi, g):
            etype, _, eidxs, _, _, soln_op, _ = g
            if snap.grad_data is None:
                return None
            gd = snap.grad_data.get(etype)
            if gd is None:
                return None

            cg = np.einsum('sp,dpvn->dsvn', soln_op, gd[..., eidxs])
            return snap.to_grad_pris(cons[gi], cg.transpose(2, 0, 1, 3), cfg)

        raw_grad_pris = self._assemble(gp)
        self._finalize_sample(sample, snap, raw_pris, raw_grad_pris)

        # Aux pass-through
        aux_pieces = defaultdict(lambda: defaultdict(list))
        for etype, _, eidxs, itype, *_ in self._groups:
            for name, arr in snap.aux(etype).items():
                aux_pieces[itype][name].append(arr[eidxs])
        for itype, by_name in aux_pieces.items():
            for name, pieces in by_name.items():
                sample.field_arrays[itype, name] = np.concatenate(pieces)

    def cell_curved(self, itype):
        parts = [self.mesh.spts_curved[g[0]][g[2]]
                 for g in self._groups if g[3] == itype]
        return np.concatenate(parts) if parts else np.empty(0, dtype=bool)


class PointsSnapshotRegion(BaseSnapshotRegion):
    def __init__(self, mesh, cfg, ppts):
        super().__init__(mesh, cfg)

        self.etypes = ['points']
        self.clean = False

        _, rank, root = get_comm_rank_root()
        self._is_root = rank == root

        self._ppts = np.asarray(ppts, dtype=float)

        self._build_geometry()

    def _build_geometry(self):
        locs = PointLocator(self.mesh).locate(self._ppts)
        self._sampler = PointSampler(self.mesh, self._ppts, locs)

        if self._is_root:
            self.ploc = {'points': np.ascontiguousarray(self._ppts.T)}
        else:
            self.ploc = {'points': np.empty((self.ndims, 0))}

    def _configure_sampler(self, snap):
        nvars = len(snap.pris_names)
        nfields = nvars*(1 + self.ndims) if snap.has_grads else nvars
        self._sampler.configure_with_cfg_nvars(self.config, nfields)

    def _compute_sample(self, sample, snap):
        self._ensure_form(snap, lambda: self._configure_sampler(snap))
        nvars = len(snap.pris_names)
        has_grads = bool(snap.has_grads)

        data = []
        for et in self.mesh.eidxs:
            s = snap.data[et]
            if has_grads:
                g = snap.grad_data[et]
                gflat = g.transpose(1, 2, 0, 3).reshape(
                    s.shape[0], -1, s.shape[2])
                s = np.concatenate([s, gflat], axis=1)
            data.append(s)

        samps = self._sampler.sample(data)
        sample._samps = samps.swapaxes(0, 1) if samps.size else samps

        cfg = self.config
        if self._is_root:
            primary = sample._samps[:nvars]
            sample.pris = {'points': snap.to_pris(primary, cfg)}

            if has_grads:
                cg = sample._samps[nvars:].reshape(nvars, self.ndims, -1)
                sample.grad_pris = {'points': snap.to_grad_pris(primary, cg,
                                                                cfg)}
            else:
                sample.grad_pris = {'points': None}

            self._publish_data_columns(sample, snap, nvars, has_grads)
        else:
            sample.pris = {'points': [np.empty(0) for _ in range(nvars)]}
            if has_grads:
                sample.grad_pris = {'points': [np.empty((self.ndims, 0))
                                               for _ in range(nvars)]}
            else:
                sample.grad_pris = {'points': None}

    def _publish_data_columns(self, sample, snap, nvars, has_grads):
        samps = sample.samples
        for info in snap.fields.values():
            if info.source == 'data':
                sample.field_arrays['points', info.name] = \
                    samps[:, info.data_index]
            elif info.source == 'grad_data' and has_grads:
                lo = nvars + info.data_index*self.ndims
                hi = lo + self.ndims
                sample.field_arrays['points', info.name] = samps[:, lo:hi]
