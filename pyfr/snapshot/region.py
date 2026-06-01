from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.points import PointLocator, PointSampler
from pyfr.shapes import BaseShape, proj_pts
from pyfr.snapshot.base import _interp
from pyfr.snapshot.sample import SnapshotSample
from pyfr.subdiv import CleanToGrid
from pyfr.util import subclass_where


# ---------------------------------------------------------------------------
# Regions: long-lived geometry (built once per plugin / per file).  No
# solution data, no intg reference, no snap reference.  Each region knows how
# to compute a SnapshotSample from a (transient) snap via _compute_sample().
# ---------------------------------------------------------------------------

class BaseSnapshotRegion:
    # Base class for all region kinds.  A region holds geometry-only state
    # (interpolation operators, CleanToGrid topology, ploc) captured once
    # from a `snap` at construction time.  Subclasses (VolumeSnapshotRegion,
    # SurfaceSnapshotRegion, PointsSnapshotRegion) own their specific
    # geometry and sample-computation logic.
    #
    # Construction contract:
    #   * Subclass __init__ calls super().__init__(snap) first to capture the
    #     static metadata (mesh, config, elementscls, ndims, dtype) — no snap
    #     reference is retained.
    #   * Subclass then stores its own region-specific parameters and calls
    #     self._build_geometry() to materialise geometry.

    def __init__(self, snap):
        self.mesh = snap.mesh
        self.config = snap.config
        self.elementscls = snap.elementscls
        self.ndims = snap.ndims
        self.dtype = snap.dtype

    def sample(self, snap):
        # Factory: build a SnapshotSample for this region against `snap`.
        return SnapshotSample(self, snap)

    def _build_geometry(self):
        raise NotImplementedError

    def _compute_sample(self, sample, snap):
        raise NotImplementedError

    def _finalize_sample(self, sample, raw_pris, raw_grad_pris):
        # Apply the cleaner dedup when clean=True; pass raw layouts through
        # otherwise.  Shared between volume + surface (Points doesn't clean).
        # grad_pris may legitimately be all-None (no gradients in the snap),
        # in which case we keep the raw layout regardless of clean.
        if self.clean:
            sample.pris = self._clean_pris(raw_pris)
            if any(v is None for v in raw_grad_pris.values()):
                sample.grad_pris = raw_grad_pris
            else:
                sample.grad_pris = self._clean_grad_pris(raw_grad_pris)
        else:
            sample.pris = raw_pris
            sample.grad_pris = raw_grad_pris

    def _clean_pris(self, raw):
        # Batch all keys through one cleaner.average so dedup spans shared
        # sub-points across element/face types.  Empty ranks pass empty
        # stacked + nvars=0 locally; the Allreduce'd nvars and self.dtype make
        # the call shape-compatible with peers.
        from pyfr.mpiutil import mpi
        comm, _, _ = get_comm_rank_root()

        stacked = {k: np.stack(prims_list, axis=-1)      # (..., nvars)
                   for k, prims_list in raw.items()}
        nvars = comm.allreduce(
            next((len(p) for p in raw.values()), 0), op=mpi.MAX)

        avg = self.cleaner.average(stacked, nvars, self.dtype)
        return {k: list(avg[k].T) for k in raw}

    def _clean_grad_pris(self, raw):
        # Same batched dedup as _clean_pris but with an extra ndims axis:
        # flatten (nvars, ndims) → single ncomp axis for the cleaner, then
        # unflatten.  Empty-rank coordination via Allreduce.
        from pyfr.mpiutil import mpi
        comm, _, _ = get_comm_rank_root()

        flat, nvars, ndims = {}, 0, 0
        for k, grads_list in raw.items():
            stacked = np.stack(grads_list, axis=0)       # (nvars,ndims,...)
            nvars, ndims = stacked.shape[:2]
            flat[k] = stacked.reshape(nvars*ndims, *stacked.shape[2:]
                                      ).transpose(1, 2, 0)

        nvars = comm.allreduce(nvars, op=mpi.MAX)
        ndims = comm.allreduce(ndims, op=mpi.MAX)

        avg = self.cleaner.average(flat, nvars*ndims, self.dtype)
        return {k: [c for c in avg[k].T.reshape(nvars, ndims, -1)] for k in raw}


class VolumeSnapshotRegion(BaseSnapshotRegion):
    # The general-purpose region.  A point set sampled from each requested
    # element type at reference points chosen by `refpts_fn(shapecls, shape)`.
    # Produced by Snapshot.region() / Snapshot.vis() / direct construction.
    #
    # `spec` is a dict mapping etype -> eidxs (the geometric subset of
    # elements per etype, as produced by region_data()).  eidxs=None for an
    # etype means "all elements of that type."  The simpler list-of-etypes
    # form (`['hex', 'tet']`) is also accepted as shorthand for the all-
    # elements case.  Snapshot.region() / Snapshot.vis() normalise both to
    # the dict form before passing in.

    def __init__(self, snap, spec, refpts_fn, *, divisor=None, clean=False):
        super().__init__(snap)

        # Normalise spec to {etype: eidxs|None}.  region_data() returns
        # `slice(None)` for the all-elements case (and integer arrays for
        # real subsets), so map any slice → None (meaning "all elements")
        # and array-ify the rest.
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
        # mesh.spts[et]: (nspts, neles, ndims) — optionally subset by eidxs.
        spts = self.mesh.spts[et]
        eidxs = self._eidxs[et]
        return spts if eidxs is None else spts[:, eidxs]

    def _ele_spts_nodes(self, et):
        # mesh.spts_nodes[et]: (neles, nnodes_per_ele) — optionally subset.
        spts_nodes = self.mesh.spts_nodes[et]
        eidxs = self._eidxs[et]
        return spts_nodes if eidxs is None else spts_nodes[eidxs]

    def _ele_soln(self, snap, et):
        # snap.soln(et): (nupts, nvars, neles) — optionally subset on neles.
        soln = snap.soln(et)
        eidxs = self._eidxs[et]
        return soln if eidxs is None else soln[..., eidxs]

    def _ele_grad_soln(self, snap, et):
        # snap.grad_soln(et): (ndims, nupts, nvars, neles) | None — subset.
        g = snap.grad_soln(et)
        if g is None:
            return None
        eidxs = self._eidxs[et]
        return g if eidxs is None else g[..., eidxs]

    def _build_geometry(self):
        # Per-etype interpolation ops keyed by etype: (mesh_op, soln_op).  The
        # raw refpts are also cached for the cleaner (and for downstream
        # consumers that want to know the underlying sampling).
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

        # Build the cleaner if clean=True.  CleanToGrid.__init__ contains
        # Allgather + Allreduce, so every rank must hit it.
        self.cleaner = self._build_cleaner() if self.clean else None

        # Physical coords at the sample points (geometry-only)
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
            xd = _interp(mesh_op, self._ele_spts(et))   # (nsvpts,neles,ndims)
            if self.clean:
                out[et] = np.ascontiguousarray(self.cleaner.select(et, xd).T)
            else:
                out[et] = np.ascontiguousarray(xd.transpose(2, 0, 1))
        return out

    def _compute_sample(self, sample, snap):
        # Called by SnapshotSample.__init__ to fill pris / grad_pris from
        # `snap`'s soln data using this region's interpolation ops + cleaner.
        # The form-specific transform (con_to_pri vs identity) is delegated
        # to snap.to_pris / snap.to_grad_pris — region stays form-agnostic.
        # Every rank calls this in lock-step (per-call from the renderer),
        # so the cleaner collective inside fires uniformly.
        cfg = self.config

        # Interpolate stored data to refpts: (nrefpts, nvars, neles).
        interp_data = {et: _interp(self._ops[et][1],
                                   self._ele_soln(snap, et)).swapaxes(0, 1)
                       for et in self.etypes}

        raw_pris = {et: snap.to_pris(interp_data[et], cfg)
                    for et in self.etypes}

        raw_grad_pris = {}
        for et in self.etypes:
            g = self._ele_grad_soln(snap, et)
            if g is None:
                raw_grad_pris[et] = None
                continue

            _, soln_op = self._ops[et]
            cg = np.einsum('sp,dpvn->dsvn', soln_op, g)
            raw_grad_pris[et] = snap.to_grad_pris(
                interp_data[et], cg.transpose(2, 0, 1, 3), cfg)

        self._finalize_sample(sample, raw_pris, raw_grad_pris)

        # Aux pass-through, sliced by the region's eidxs subset.  Aux arrays
        # have neles on axis 0 — slice on axis 0 to match the region's element
        # subset; remaining axes pass through unchanged.
        for et in self.etypes:
            eidxs = self._eidxs[et]
            for name, arr in snap.aux(et).items():
                sample.fields[et, name] = arr if eidxs is None else arr[eidxs]


class SurfaceSnapshotRegion(BaseSnapshotRegion):
    # A boundary region.  Faces on the named boundary(s), keyed by face element
    # type (itype) for the per-call sample's pris/grad_pris.  Adds geometry-
    # only normals + wall_dist on top of the base ops/cleaner/ploc.

    def __init__(self, snap, bcname, divisor, refpts_fn=None, clean=True):
        super().__init__(snap)

        # Region parameters
        self._divisor = divisor
        self._refpts_fn = (refpts_fn or
                           (lambda sc, _sh=None: sc.std_ele(self._divisor)))
        self.clean = clean

        bcnames = [bcname] if isinstance(bcname, str) else list(bcname)

        # Merge the connectivity from each boundary into one (etype, fidx) ->
        # eidxs map; partitioned meshes may not have any face from a given
        # boundary on every rank — empty region on that rank is fine.
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
        # One entry per boundary face-group:
        #     (etype, fidx, eidxs, itype, mesh_op, soln_op, fpts)
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

        # Backend-less element instance per face-group, for geometry queries
        # (normals, wall-dist).  Aligned with _groups.
        self._eles = []
        for etype, _, eidxs, *_ in self._groups:
            shapecls = subclass_where(BaseShape, name=etype)
            spts = self.mesh.spts[etype][:, eidxs]
            self._eles.append(self.elementscls(shapecls, spts, self.config))

        # Cleaner — collective; built once on every rank
        self.cleaner = self._build_cleaner() if self.clean else None

        # Geometry-only quantities (depend on mesh + face ops, not the soln)
        self.ploc = self._build_ploc()
        self.normals = self._build_normals()
        self.wall_dist = self._build_wall_dist()

    def _assemble(self, fn):
        # Map fn(gi, group) over face-groups, concatenating same-itype results
        # along the element axis.
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
            xd = _interp(mesh_op, self.mesh.spts[etype][:, eidxs])
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
        # Per-face-group sampling then itype-wise assemble + clean.  Form
        # dispatch is delegated to snap.to_pris / snap.to_grad_pris.
        cfg = self.config

        cons = []
        for etype, _, eidxs, _, _, soln_op, _ in self._groups:
            c = _interp(soln_op, snap.soln(etype)[:, :, eidxs])
            cons.append(c.swapaxes(0, 1))

        raw_pris = self._assemble(lambda gi, g: snap.to_pris(cons[gi], cfg))

        def gp(gi, g):
            etype, _, eidxs, _, _, soln_op, _ = g
            gd = snap.grad_soln(etype)
            if gd is None:
                return None

            cg = np.einsum('sp,dpvn->dsvn', soln_op, gd[..., eidxs])
            return snap.to_grad_pris(cons[gi], cg.transpose(2, 0, 1, 3), cfg)

        raw_grad_pris = self._assemble(gp)
        self._finalize_sample(sample, raw_pris, raw_grad_pris)

        # Aux pass-through, per face-group: slice the volume etype's aux by
        # the group's eidxs, concat per itype.  Matches the existing boundary
        # writer's _cell_aux convention (aux array's first axis is neles).
        aux_pieces = defaultdict(lambda: defaultdict(list))
        for etype, _, eidxs, itype, *_ in self._groups:
            for name, arr in snap.aux(etype).items():
                aux_pieces[itype][name].append(arr[eidxs])
        for itype, by_name in aux_pieces.items():
            for name, pieces in by_name.items():
                sample.fields[itype, name] = np.concatenate(pieces)


class PointsSnapshotRegion(BaseSnapshotRegion):
    # A region sampled at user-supplied physical points.  Geometry lives in
    # the PointLocator results + PointSampler config (built once); per-call
    # sampling re-runs sampler.sample() against the current snap.
    #
    # Layout is flat under a single pseudo-etype 'points':
    #     ploc['points']      -> (ndims, npts)
    #     pris['points']      -> [nvars (npts,)]
    #     grad_pris['points'] -> [nvars (ndims, npts)] | None
    #
    # MPI-aware: PointSampler gathers samples on the root rank; non-root
    # ranks see empty arrays.
    def __init__(self, snap, ppts):
        super().__init__(snap)

        self.etypes = ['points']
        self.clean = False

        _, rank, root = get_comm_rank_root()
        self._is_root = rank == root

        self._ppts = np.asarray(ppts, dtype=float)
        # _nvars works for both forms: pris_names is privars for soln,
        # stored_fields for stats.  Grad availability is also form-aware —
        # StatsSnapshot fixes has_grads=False at the class level.
        self._nvars = len(snap.pris_names)
        self._has_grads = snap.has_grads

        self._build_geometry()

    def _build_geometry(self):
        nfields = (self._nvars*(1 + self.ndims) if self._has_grads
                   else self._nvars)

        locs = PointLocator(self.mesh).locate(self._ppts)
        self._sampler = PointSampler(self.mesh, self._ppts, locs)
        self._sampler.configure_with_cfg_nvars(self.config, nfields)

        # ploc is just the user-supplied points (on root only)
        if self._is_root:
            self.ploc = {'points': np.ascontiguousarray(self._ppts.T)}
        else:
            self.ploc = {'points': np.empty((self.ndims, 0))}

    def _compute_sample(self, sample, snap):
        # Re-sample at the user-supplied points and rebuild pris + grad_pris.
        # PointSampler does an MPI gather to root; every rank participates.
        data = []
        for et in self.mesh.eidxs:
            s = snap.soln(et)                               # (nu, nvars, ne)
            if self._has_grads:
                g = snap.grad_soln(et)                      # (nd, nu, nvars, ne)
                gflat = g.transpose(1, 2, 0, 3).reshape(
                    s.shape[0], -1, s.shape[2])             # (nu, nvars*nd, ne)
                s = np.concatenate([s, gflat], axis=1)
            data.append(s)

        samps = self._sampler.sample(data)
        sample._samps = samps.swapaxes(0, 1) if samps.size else samps

        cfg = self.config
        if self._is_root:
            primary = sample._samps[:self._nvars]
            sample.pris = {'points': snap.to_pris(primary, cfg)}

            if self._has_grads:
                cg = sample._samps[self._nvars:].reshape(self._nvars,
                                                         self.ndims, -1)
                sample.grad_pris = {'points': snap.to_grad_pris(primary, cg,
                                                                cfg)}
            else:
                sample.grad_pris = {'points': None}
        else:
            sample.pris = {'points': [np.empty(0)
                                      for _ in range(self._nvars)]}
            if self._has_grads:
                sample.grad_pris = {'points': [np.empty((self.ndims, 0))
                                               for _ in range(self._nvars)]}
            else:
                sample.grad_pris = {'points': None}
