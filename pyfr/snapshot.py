from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.points import PointLocator, PointSampler
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape, proj_pts
from pyfr.subdiv import CleanToGrid
from pyfr.util import subclass_where


def _interp(op, arr):
    # op (m, k); arr (k, ...) -> (m, ...)
    return (op @ arr.reshape(arr.shape[0], -1)).reshape(op.shape[0],
                                                        *arr.shape[1:])


class Snapshot:
    # A coherent (mesh + solution + config + state) view of a dataset.  Backings:
    #   IntgSnapshot(intg)              — wraps a running integrator.  Transient:
    #                                     plugin __call__(intg) builds a fresh
    #                                     snap per step and drops it.
    #   FileSnapshot(meshf, solnf)      — wraps a .pyfrm + .pyfrs pair.
    #
    # Static surface every backing exposes as plain attributes:
    #   mesh, config, elementscls, ele_types, dtype, has_grads
    #
    # Per-step surface (live: passthrough to intg; file: set once on load):
    #   tcurr, cycle, state
    #
    # Solution access methods:
    #   soln(etype)      -> (nupts, nvars, neles)        conservative
    #   grad_soln(etype) -> (ndims, nupts, nvars, neles) or None

    @property
    def ndims(self):
        return self.mesh.ndims

    # --- region factories (geometry-only; build a sample from the region) ---
    def region(self, spec='*', refpts_fn=None):
        # Generic: a region sampled at reference points chosen by refpts_fn.
        # Default refpts_fn returns the element's native solution points.
        etypes = self.ele_types if spec == '*' else spec
        rfn = refpts_fn or (lambda sc, sh: sh.upts)
        return VolumeSnapshotRegion(self, etypes, rfn)

    def vis(self, spec='*', divisor=None, *, clean=True):
        # A vis-derived region: sample points are the subdivision of each
        # element at `divisor`.  clean=True (default) deduplicates coincident
        # sub-points and averages primitives/gradients at shared positions.
        etypes = self.ele_types if spec == '*' else spec
        div = divisor if divisor is not None else self.config.getint('solver',
                                                                     'order')
        refpts_fn = lambda sc, sh: sc.std_ele(div)
        return VolumeSnapshotRegion(self, etypes, refpts_fn, divisor=div,
                                    clean=clean)

    def surface(self, name, divisor=None, refpts_fn=None, *, clean=True):
        # A boundary region: faces on the named boundary, plus geometric
        # normals + wall distance.  `name` may be a single bcname or a list
        # (faces from each are concatenated by itype).
        div = divisor if divisor is not None else self.config.getint('solver',
                                                                     'order')
        return SurfaceSnapshotRegion(self, name, div, refpts_fn, clean=clean)

    def at_points(self, ppts):
        # A region sampled at arbitrary physical points (PointLocator finds the
        # owning element + reference coords; PointSampler evaluates soln/grad
        # there).  MPI-aware: PointSampler gathers on root.
        return PointsSnapshotRegion(self, ppts)

    def fields(self, names, on=None):
        # Convenience: build a sample, run the named field providers on it, and
        # return the populated sample.  Defaults to the full native-point view.
        region = on if on is not None else self.region()
        sample = region.sample(self)
        sample.run(names)
        return sample


class IntgSnapshot(Snapshot):
    # Wraps a running integrator.  Built transiently — the plugin's
    # __call__(intg) creates a fresh IntgSnapshot per step and drops it.  No
    # intg reference is retained by anything else (renderer/region hold their
    # own static metadata; the snap is passed into render(snap) per step).
    def __init__(self, intg):
        sys = intg.system

        # Static surface
        self.mesh = sys.mesh
        self.config = intg.cfg
        self.elementscls = sys.elementscls
        self.ele_types = list(sys.ele_types)
        self.dtype = sys.backend.fpdtype
        self.has_grads = True

        # Per-step surface (read fresh from intg each access)
        self._intg = intg

    @property
    def tcurr(self):
        return self._intg.tcurr

    @property
    def cycle(self):
        return self._intg.nacptsteps

    @property
    def state(self):
        # Live serialisable plugin/kinematic records, keyed by sprefix
        return {p.sprefix: p._serialise_data()
                for p in self._intg.plugins
                if getattr(p, 'sprefix', None)
                and getattr(p, '_serialise_data', None) is not None}

    def soln(self, etype):
        return self._intg.soln[self.ele_types.index(etype)]

    def grad_soln(self, etype):
        # intg.grad_soln is already (ndims, nupts, nvars, neles) per etype,
        # matching SolnSnapshot.grad_soln — return directly.
        return self._intg.grad_soln[self.ele_types.index(etype)]


class SolnSnapshot(Snapshot):
    # Wraps an already-loaded NativeSoln (mesh + soln + cfg + elementscls).
    # Used by consumers that opened a .pyfrs themselves (e.g. VTK writers
    # holding their own NativeReader for raw-HDF5 access) and want a Snapshot
    # interface over the same in-memory data — no double-read.  Immutable
    # for the snap's lifetime; the entire surface is plain attributes.
    def __init__(self, mesh, soln, cfg, elementscls, *, meshf=None,
                 pname=None):
        # File paths optional — kept for surface()'s lazy bcon re-read when
        # the caller can point at the source .pyfrm.
        self._meshf, self._pname = meshf, pname
        self._soln = soln

        prec = cfg.get('backend', 'precision', 'single')
        stats = soln.stats

        self.mesh = mesh
        self.config = cfg
        self.stats = stats
        self.elementscls = elementscls
        self.ele_types = list(soln.data)
        self.dtype = np.float32 if prec == 'single' else np.float64
        self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
        self.cycle = stats.getint('solver-time-integrator', 'nacptsteps', 0)
        self.has_grads = bool(soln.grad_data)
        self.state = soln.state
        # Stored field names as they appear in the file — convars for soln-
        # prefix files, tavg/residual names for those.
        self.stored_fields = list(soln.fields)

    def soln(self, etype):
        return self._soln.data[etype]

    def grad_soln(self, etype):
        return self._soln.grad_data.get(etype)

    def surface(self, name, divisor=None, refpts_fn=None, *, clean=True):
        # Boundary connectivity isn't built for cheap volume access; build it
        # lazily on first surface request — only possible when meshf was
        # supplied at construction.
        if not self.mesh.bcon and self._meshf:
            self.mesh = NativeReader(self._meshf, self._pname,
                                     construct_con=True).mesh

        return super().surface(name, divisor, refpts_fn, clean=clean)


class FileSnapshot(SolnSnapshot):
    # SolnSnapshot constructed by opening a .pyfrm mesh + .pyfrs solution file
    # from disk.  Common offline entry point used by the CLI tools.
    def __init__(self, meshf, solnf, pname=None, construct_con=False):
        # Inline: pyfr.solvers transitively pulls in pyfr.integrators which
        # imports get_plugin from pyfr.plugins; hoisting this would cycle when
        # snapshot.py is loaded during pyfr.plugins.__init__.
        from pyfr.solvers.base import BaseSystem

        reader = NativeReader(meshf, pname, construct_con=construct_con)
        mesh, soln = reader.load_subset_mesh_soln(solnf)
        cfg = soln.config

        syscls = subclass_where(BaseSystem, name=cfg.get('solver', 'system'))
        super().__init__(mesh, soln, cfg, syscls.elementscls,
                         meshf=meshf, pname=pname)


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

    def __init__(self, snap, etypes, refpts_fn, *, divisor=None, clean=False):
        super().__init__(snap)

        # Region parameters
        self.etypes = list(etypes)
        self._refpts_fn = refpts_fn
        self._divisor = divisor
        self.clean = clean

        self._build_geometry()

    def _build_geometry(self):
        # Per-etype interpolation ops keyed by etype: (mesh_op, soln_op).  The
        # raw refpts are also cached for the cleaner (and for downstream
        # consumers that want to know the underlying sampling).
        self._ops = {}
        self._refpts_at = {}
        for et in self.etypes:
            shapecls = subclass_where(BaseShape, name=et)
            spts = self.mesh.spts[et]
            shape = shapecls(spts.shape[0], self.config)
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
            spts_nodes = mesh.spts_nodes[et]
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
            xd = _interp(mesh_op, self.mesh.spts[et])    # (nsvpts,neles,ndims)
            if self.clean:
                out[et] = np.ascontiguousarray(self.cleaner.select(et, xd).T)
            else:
                out[et] = np.ascontiguousarray(xd.transpose(2, 0, 1))
        return out

    def _compute_sample(self, sample, snap):
        # Called by SnapshotSample.__init__ to fill pris / grad_pris from
        # `snap`'s soln data using this region's interpolation ops + cleaner.
        # Every rank calls this in lock-step (per-call from the renderer),
        # so the cleaner collective inside fires uniformly.
        ecls, cfg = self.elementscls, self.config

        # Sampled conservative — shared by pris and grad_pris below
        cons = {et: _interp(self._ops[et][1], snap.soln(et)).swapaxes(0, 1)
                for et in self.etypes}

        # Raw pris + grad_pris (no dedup yet)
        raw_pris = {et: list(ecls.con_to_pri(cons[et], cfg))
                    for et in self.etypes}

        raw_grad_pris = {}
        for et in self.etypes:
            g = snap.grad_soln(et)
            if g is None:
                raw_grad_pris[et] = None
                continue

            _, soln_op = self._ops[et]
            cg = np.einsum('sp,dpvn->dsvn', soln_op, g)
            raw_grad_pris[et] = list(ecls.grad_con_to_pri(
                cons[et], cg.transpose(2, 0, 1, 3), cfg))

        self._finalize_sample(sample, raw_pris, raw_grad_pris)


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
        # Per-face-group sampling then itype-wise assemble + clean.
        ecls, cfg = self.elementscls, self.config

        cons = []
        for etype, _, eidxs, _, _, soln_op, _ in self._groups:
            c = _interp(soln_op, snap.soln(etype)[:, :, eidxs])
            cons.append(c.swapaxes(0, 1))

        raw_pris = self._assemble(
            lambda gi, g: list(ecls.con_to_pri(cons[gi], cfg)))

        def gp(gi, g):
            etype, _, eidxs, _, _, soln_op, _ = g
            gd = snap.grad_soln(etype)
            if gd is None:
                return None

            cg = np.einsum('sp,dpvn->dsvn', soln_op, gd[..., eidxs])
            return list(ecls.grad_con_to_pri(cons[gi],
                                             cg.transpose(2, 0, 1, 3), cfg))

        raw_grad_pris = self._assemble(gp)
        self._finalize_sample(sample, raw_pris, raw_grad_pris)


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
        self._nvars = len(snap.elementscls.privars(snap.ndims, snap.config))
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

        ecls, cfg = self.elementscls, self.config
        if self._is_root:
            cons = sample._samps[:self._nvars]
            sample.pris = {'points': list(ecls.con_to_pri(cons, cfg))}

            if self._has_grads:
                cg = sample._samps[self._nvars:].reshape(self._nvars,
                                                         self.ndims, -1)
                sample.grad_pris = {'points': list(
                    ecls.grad_con_to_pri(cons, cg, cfg))}
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


# ---------------------------------------------------------------------------
# SnapshotSample: solution evaluated on a region's geometry for one snap.
# Per-call object; created by region.sample(snap), used for one render step
# (or one offline export pass), then dropped.  Holds:
#   - ploc: starts as an ALIAS to region.ploc; transformer providers (NIRF)
#     call make_ploc_writable() to copy-on-write before mutating, so the
#     region's static body-frame geometry is never compounded across calls.
#   - pris, grad_pris: per-call (populated by region._compute_sample(...))
#   - fields: provider outputs keyed by (etype, fname)
# ---------------------------------------------------------------------------

class SnapshotSample:
    def __init__(self, region, snap):
        self.region = region
        self.snap = snap

        # Provider outputs land here, keyed by (etype/itype, field name)
        self.fields = {}

        # ploc starts aliased to the region's static coords; a transformer
        # provider calls make_ploc_writable() before mutating.
        self.ploc = region.ploc

        # Region populates pris + grad_pris into self via this call.
        region._compute_sample(self, snap)

    def make_ploc_writable(self):
        # Copy-on-write: ensure self.ploc is independent of region.ploc before
        # a transformer mutates it.  After the sample is dropped, the copy is
        # gone and the next sample built from this region starts fresh.
        if self.ploc is self.region.ploc:
            self.ploc = {k: v.copy() for k, v in self.ploc.items()}

    @property
    def samples(self):
        # Tabular accessor for sampled conservative + (optional) grads on
        # PointsSnapshotRegion-built samples.  Layout: (npts, nfields) on root,
        # empty on non-root.  AttributeError elsewhere — only at_points()
        # regions populate the underlying _samps via their _compute_sample.
        samps = self._samps
        return samps.T if samps.size else samps

    def run(self, names_or_runner, *, export_type='volume', public_only=True,
            cfg=None):
        # Run field providers on this sample.  Each provider gets a per-key
        # SampleView lens; reads pris/grad_pris/normals/state/etc., writes
        # into view.fields (collected back into self.fields).  Transformer
        # providers mutate ploc/pris in place via the view.
        #
        # `names_or_runner` may be a list of provider names or a pre-built
        # FieldRunner — pass the runner directly to skip topo-sort + dep
        # resolution in hot paths (the in-situ renderer caches its runners
        # at init).  `cfg` overrides the snapshot's config when building from
        # names (used by cli/sampler --cfg pp.ini).
        if hasattr(names_or_runner, 'run_on_sample'):
            runner = names_or_runner
        else:
            # Inline import: pyfr.plugins.fields.runner lives under pyfr.plugins
            # whose __init__ would otherwise try to load snapshot.py back
            # before plugins is fully initialised.
            from pyfr.plugins.fields.runner import FieldRunner
            runner = FieldRunner(names_or_runner, self.region.ndims,
                                 cfg or self.snap.config, export_type)

        return runner.run_on_sample(self, public_only=public_only)

    def flat(self, name):
        # Unified all-points view that hides the etype split, for scripting:
        #   sample.flat('ploc')   -> (ndims, N)
        #   sample.flat('pris')   -> (nvars, N)
        #   sample.flat('mach')   -> (N,)
        etypes = self.region.etypes

        def cat(arrs, nlead):
            return np.concatenate(
                [a.reshape(*a.shape[:nlead], -1) for a in arrs], axis=-1)

        if name == 'ploc':
            return cat([self.ploc[et] for et in etypes], 1)
        elif name == 'pris':
            byet = self.pris
            nvars = len(byet[etypes[0]])
            return np.stack([cat([byet[et][i] for et in etypes], 0)
                             for i in range(nvars)])
        elif name == 'grad_pris':
            byet = self.grad_pris
            nvars = len(byet[etypes[0]])
            return np.stack([cat([byet[et][i] for et in etypes], 1)
                             for i in range(nvars)])
        else:
            return cat([self.fields[et, name] for et in etypes], 0)
