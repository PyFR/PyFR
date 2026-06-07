import re
from collections import defaultdict, namedtuple

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import (AlltoallMixin, get_comm_rank_root, home_rank, mpi,
                          scal_coll)
from pyfr.nputil import range_offsets, search_unsorted
from pyfr.points import PointLocator
from pyfr.polys import get_polybasis
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape, proj_pts
from pyfr.snapshot.region import BaseSnapshotRegion, interp_ops
from pyfr.subdiv import CleanToGrid, NullCleaner
from pyfr.util import subclass_where


class _SpanwiseBase:
    def __init__(self, mesh, cfg, shapes, *, axis=2, vis_pts_2d=None):
        self.mesh = mesh
        self.pmesh = mesh.parent or mesh
        self.cfg = cfg
        self.shape = shapes

        self.axis = axis
        self.in_axes = [i for i in range(3) if i != axis]
        self.order = cfg.getint('solver', 'order')
        self.vis_pts_2d = vis_pts_2d

    @memoize
    def _sample_pts_2d(self, itype):
        ishapecls = subclass_where(BaseShape, name=itype)
        svpts = ishapecls.std_ele(self.order)
        c = ishapecls.std_ele(1).mean(axis=0)
        return c + 0.95*(svpts - c)

    @memoize
    def _vis_op(self, itype):
        basis = get_polybasis(itype, self.order, self._sample_pts_2d(itype))
        return basis.nodal_basis_at(self.vis_pts_2d[itype])

    def span_extent(self):
        comm, _, _ = get_comm_rank_root()
        lo, hi = float('inf'), float('-inf')
        for spts in self.mesh.spts.values():
            lo = min(lo, spts[..., self.axis].min())
            hi = max(hi, spts[..., self.axis].max())

        return (scal_coll(comm.Allreduce, lo, op=mpi.MIN),
                scal_coll(comm.Allreduce, hi, op=mpi.MAX))


_EInfo = namedtuple('_EInfo', 'bidx cnidx spanop shapeop cdtype')


class ExtrudedSpanwise(_SpanwiseBase, AlltoallMixin):
    def __init__(self, mesh, cfg, shapes, *, nfields, vis_pts_2d, axis=2):
        super().__init__(mesh, cfg, shapes, axis=axis, vis_pts_2d=vis_pts_2d)

        if set(self.pmesh.etypes) - {'hex', 'pri'}:
            raise ValueError(f'Extruded spanwise needs a hex/pri mesh')

        self.nfields = nfields

        # Construct the topology and operators for each element type
        self.einfo = {et: self._build_etype(et) for et in self.pmesh.etypes
                      if et in ('hex', 'pri')}

        # Pre-compute the global span length (single Allreduce)
        slo, shi = self.span_extent()
        self.span_len = shi - slo

        # Pre-compute per-element column ids
        self.col_ids = self._compute_col_ids()

    def _build_etype(self, et):
        itype = 'quad' if et == 'hex' else 'tri'
        sptord = self.shape[et].nsptsord
        ra, ria = (2, [0, 1]) if et == 'pri' else (self.axis, self.in_axes)

        ishapecls = subclass_where(BaseShape, name=itype)
        spts2d = ishapecls.std_ele(sptord)

        # Bottom/top spts in reference space
        std = self.shape[et].std_ele(sptord)
        z = std[:, ra]
        bot = np.flatnonzero(np.isclose(z, z.min(), atol=1e-9))
        top = np.flatnonzero(np.isclose(z, z.max(), atol=1e-9))

        # Validate extrusion; in-plane coords must match top to bottom
        p = self.pmesh.spts[et][..., self.in_axes]
        err = np.abs(p[top] - p[bot]).max(axis=(0, 2))
        ext = np.linalg.norm(np.ptp(p[bot], axis=0), axis=-1)
        if (err > 1e-2*ext).any():
            raise ValueError(f'{et} extrusion is not flat')

        # Bottom spts reordered to match 2D output convention
        diff = std[bot][:, ria, None] - spts2d.T[None]
        bidx = bot[np.argmin(np.linalg.norm(diff, axis=1), axis=0)]
        cnidx = bidx[ishapecls.corner_pts_idxs(len(spts2d))]

        # Intermediate points
        pts2d = self._sample_pts_2d(itype)
        pts1d = np.linspace(-1, 1, self.order + 1)
        eq3d = np.empty((len(pts2d)*len(pts1d), 3))
        eq3d[:, ria] = np.tile(pts2d, (len(pts1d), 1))
        eq3d[:, ra] = np.repeat(pts1d, len(pts2d))

        # Fused interpolation + span-averaging operator
        eqop = self.shape[et].ubasis.nodal_basis_at(eq3d)
        swts = 0.5*2**0.5*get_polybasis('line', self.order, pts1d).invvdm[:, 0]
        spanop = (swts @ eqop.reshape(len(pts1d), -1)).reshape(len(pts2d), -1)

        # Interpolation operator from shape points to vis points
        sv2d = self.vis_pts_2d[itype]
        shapeop = get_polybasis(itype, sptord, spts2d).nodal_basis_at(sv2d)

        cdtype = np.dtype([
            ('key', np.int64),
            ('sum', np.float64, (len(pts2d), self.nfields)),
            ('xy', np.float64, (len(spts2d), 2)),
            ('cnodes', np.int64, (ishapecls.npts_from_order(1),))
        ])

        return _EInfo(bidx, cnidx, spanop, shapeop, cdtype)

    def average(self, soln):
        out = []
        for et, ei in self.einfo.items():
            cols = self._reduce_etype(et, soln.get(et))
            if len(cols):
                itype = 'quad' if et == 'hex' else 'tri'
                means = cols['sum'] / self.span_len
                vsoln = self._vis_op(itype) @ means
                vxy = ei.shapeop @ cols['xy']
                vpts = np.zeros(vxy.shape[:-1] + (3,))
                vpts[..., self.in_axes] = vxy
                out.append((itype, vsoln, vpts, cols['cnodes']))

        return out

    def _compute_col_ids(self):
        comm, _, _ = get_comm_rank_root()

        # Pack every owned element into a single flat int64 array
        offs, _ = range_offsets(self.pmesh.eidxs.items())
        col = np.concatenate([np.empty(0, dtype=int),
                              *self.pmesh.eidxs.values()])
        pa, pb, links = self._span_links(offs)

        while True:
            old = col.copy()

            # Local pairs; bidirectional min-reduction on the snapshot
            lo = np.minimum(col[pa], col[pb])
            np.minimum.at(col, pa, lo)
            np.minimum.at(col, pb, lo)

            # Inter-rank pairs
            bufs = [col[idx] for _, idx, _ in links]
            mpi.Request.Waitall([comm.Isendrecv_replace(b, nr, source=nr)
                                 for b, (nr, _, _) in zip(bufs, links)])
            for b, (_, idx, mk) in zip(bufs, links):
                np.minimum.at(col, idx[mk], b[mk])

            # Stop when no rank changed any column id this round
            if not scal_coll(comm.Allreduce, bool((col != old).any()),
                             op=mpi.LOR):
                break

        # Slice the full-mesh ids by each subset etype's eidx positions
        return {et: col[offs[et] + search_unsorted(self.pmesh.eidxs[et], e)]
                for et, e in self.mesh.eidxs.items()}

    def _span_links(self, offs):
        # Per-cidx tables; etype offset; whether the face is spanwise
        lhs, rhs = self.pmesh.con
        cmap = lhs.cidxmap
        cidxoff = np.zeros(max(cmap) + 1, dtype=int)
        cidxspan = np.zeros_like(cidxoff, dtype=bool)
        for cidx, (etype, fidx) in cmap.items():
            if etype in offs:
                cidxoff[cidx] = offs[etype]
                ra = 2 if etype == 'pri' else self.axis
                norm = self.shape[etype].faces[fidx][2]
                cidxspan[cidx] = norm[ra] in (1, -1)

        # Local internal pairs across spanwise faces
        m = cidxspan[lhs.cidxs] | cidxspan[rhs.cidxs]
        pa = cidxoff[lhs.cidxs[m]] + lhs.eidxs[m]
        pb = cidxoff[rhs.cidxs[m]] + rhs.eidxs[m]

        # Inter-rank spanwise face pairs
        links = [(nr, cidxoff[c.cidxs] + c.eidxs, mk)
                 for nr, c in self.pmesh.con_p.items()
                 if (mk := cidxspan[c.cidxs]).any()]

        return pa, pb, links

    def _reduce_etype(self, etype, soln_pre):
        ei = self.einfo[etype]

        if (spts := self.mesh.spts.get(etype)) is None:
            return self._exchange(np.empty(0, dtype=ei.cdtype))

        # Interpolate to equispaced 3D and integrate along span
        soln = (ei.spanop @ soln_pre).swapaxes(0, 1)

        # Column ids and per-element span lengths
        keys = self.col_ids[etype]
        ds = np.ptp(spts[..., self.axis], axis=0)
        xy = spts[ei.bidx][..., self.in_axes]
        cnodes = self.mesh.spts_nodes[etype][:, ei.cnidx]

        # Accumulate per-column sums
        ukeys, first, inverse = np.unique(keys, return_index=True,
                                          return_inverse=True)
        cols = np.zeros(len(ukeys), dtype=ei.cdtype)
        cols['key'] = ukeys
        cols['xy'] = xy.swapaxes(0, 1)[first]
        cols['cnodes'] = cnodes[first]
        np.add.at(cols['sum'], inverse, (soln*ds).transpose(2, 0, 1))

        return self._exchange(cols)

    def _exchange(self, cols):
        comm, _, _ = get_comm_rank_root()

        # Route each column to its home rank by hashed key
        home = home_rank(cols['key'], comm.size)
        hcount = np.bincount(home, minlength=comm.size)
        resp = self._alltoallcv(comm, cols[np.argsort(home)], hcount)[0]

        # Multiple senders may have contributed to the same column
        ukeys, first, inverse = np.unique(resp['key'], return_index=True,
                                          return_inverse=True)
        out = np.zeros(len(ukeys), dtype=cols.dtype)
        out['key'] = ukeys
        out['xy'] = resp['xy'][first]
        out['cnodes'] = resp['cnodes'][first]
        np.add.at(out['sum'], inverse, resp['sum'])

        return out


class SampledSpanwise(_SpanwiseBase, AlltoallMixin):
    def __init__(self, mesh, cfg, shapes, eles, *, nfields, nstations,
                 vis_pts_2d, axis=2, boundary=None, periodic=None):
        super().__init__(mesh, cfg, shapes, axis=axis, vis_pts_2d=vis_pts_2d)

        if int(nstations) < 2:
            raise ValueError('nstations must be >= 2 (trapezium rule)')

        self.boundary = boundary
        self.periodic = periodic
        self.nstations = int(nstations)
        self.nfields = nfields
        self.eles = eles

        # Span stations (nudged inwards to dodge boundary fp noise)
        slo, shi = self.span_extent()
        ds = (shi - slo)*1e-9
        self.stations = np.linspace(slo + ds, shi - ds, self.nstations)

        self.locator = PointLocator(self.mesh)

        # Per-face-group: (itype, vpts in 3D physical at vis pts, cnodes)
        self.fgroups = [self._face_group(etype, fidx, idxs, vis_pts_2d)
                        for etype, fidx, idxs in self._raw_face_groups()]

    def _face_group(self, etype, fidx, idxs, vis_pts_2d):
        itype, proj, _ = self.shape[etype].faces[fidx]
        svpts = proj_pts(proj, vis_pts_2d[itype])
        spts = proj_pts(proj, self._sample_pts_2d(itype))

        mesh_op = self.shape[etype].sbasis.nodal_basis_at(svpts)
        smesh_op = self.shape[etype].sbasis.nodal_basis_at(spts)

        vpts = interp_ops(mesh_op, self.mesh.spts[etype][:, idxs])
        spts = interp_ops(smesh_op, self.mesh.spts[etype][:, idxs])
        cnodes = self._face_cnodes(etype, fidx, idxs)

        return itype, vpts.swapaxes(0, 1), spts.swapaxes(0, 1), cnodes

    def _face_cnodes(self, etype, fidx, idxs):
        spts_nodes = self.mesh.spts_nodes[etype]
        shapecls = subclass_where(BaseShape, name=etype)
        cidxs = shapecls.face_corner_pts_idxs(fidx, spts_nodes.shape[1])
        return spts_nodes[np.ix_(idxs, cidxs)]

    def _raw_face_groups(self):
        # List of (etype, fidx, local_eidxs) face groups to average over
        if self.boundary is not None:
            return self._bc_faces(self.boundary)
        else:
            return self._periodic_faces(self.periodic)

    def average(self, soln):
        soln = {etype: data.swapaxes(0, 1) for etype, data in soln.items()}

        # Perform the sampling for each face group
        pts = [spts.reshape(-1, 3) for _, _, spts, _ in self.fgroups]
        pts2d = np.concatenate(pts or [np.empty((0, 3))])
        means = self._sample_mean(pts2d, soln, [len(p) for p in pts])

        # Interpolate to the visualisation points
        out = []
        for m, (itype, vpts, spts, cnodes) in zip(means, self.fgroups):
            m = m.reshape(*spts.shape[:2], -1)
            vsoln = interp_ops(self._vis_op(itype), m.swapaxes(0, 1))
            out.append((itype, vsoln.swapaxes(0, 1), vpts, cnodes))

        return out

    def _sample_mean(self, pts, soln, sizes):
        means = [np.zeros((n, self.nfields)) for n in sizes]
        splits = np.cumsum(sizes[:-1])

        # Trapezium weights
        w = np.full(self.nstations, 1.0 / (self.nstations - 1))
        w[0] = w[-1] = 0.5 / (self.nstations - 1)

        for s, ws in zip(self.stations, w):
            # Displace the sample points
            pts[:, self.axis] = s

            # Locate and sample the solution at each point
            locs = self.locator.locate_disjoint(pts)
            samps = self._sample(pts, locs, soln)

            # Split the samples by face group and accumulate
            for fmeans, fsamps in zip(means, np.split(samps, splits)):
                fmeans += ws*fsamps.reshape(fmeans.shape)

        return means

    def _sample(self, pts, locs, soln):
        comm, _, _ = get_comm_rank_root()

        # Route query points to their owners
        scount = np.bincount(locs['rank'], minlength=comm.size)
        sord = np.argsort(locs['rank'], kind='stable')
        inq, (rcount, _) = self._alltoallcv(comm, locs[sord], scount)

        # Interpolate the solution at each incoming tloc
        samp = np.zeros((len(inq), self.nfields))
        for etype, edata in soln.items():
            cidx = self.mesh.codec.index(f'eles/{etype}')
            mask = inq['cidx'] == cidx

            # Map global element indices to local indices
            local = search_unsorted(self.mesh.eidxs[etype], inq['eidx'][mask])

            ubasis = self.shape[etype].ubasis
            ops = ubasis.nodal_basis_at(inq['tloc'][mask], clean=False)
            samp[mask] = np.einsum('pn,nfp->pf', ops, edata[:, :, local])

        # Return the results
        rsamp, _ = self._alltoallcv(comm, samp, rcount)

        # Re-order the samples by query point index
        samps = np.empty((len(pts), self.nfields))
        samps[sord] = rsamp
        return samps

    def _bc_faces(self, name):
        smesh, rmesh = self.mesh, self.pmesh

        try:
            cidx = smesh.codec.index(f'bc/{name}')
        except ValueError:
            raise ValueError(f'Unknown boundary {name!r}')

        out = []
        for etype, einfo in self.eles.items():
            # Boundary faces of this etype as (element, face) offset pairs
            eoff, fidx = (einfo['faces']['cidx'] == cidx).nonzero()
            if not len(eoff):
                continue

            # Translate full-mesh offsets to subset-local offsets if needed
            if smesh.subset:
                if etype not in smesh.eidxs:
                    raise ValueError('Boundary not present in subset solution')

                gidx = rmesh.eidxs[etype][eoff]
                if not np.isin(gidx, smesh.eidxs[etype]).all():
                    raise ValueError('Boundary not present in subset solution')

                eoff = search_unsorted(smesh.eidxs[etype], gidx)

            # Group elements by face index (one group per face per etype)
            for f in map(int, np.unique(fidx)):
                out.append((etype, f, eoff[fidx == f]))

        return out

    def _periodic_faces(self, name):
        smesh = self.mesh

        if (pinfo := smesh.raw.get(f'periodic/{name}')) is None:
            raise ValueError(f'Unknown periodic boundary {name!r}')

        # Map each codec index to its (etype, fidx) pair
        cidxmap = {ci: (m[1], int(m[2])) for ci, c in enumerate(smesh.codec)
                   if (m := re.fullmatch(r'eles/(\w+)/face/(\d+)', c))}

        # Take the LHS of the periodic pair as the source group
        lcidx, loff = pinfo[:, 0]['cidx'], pinfo[:, 0]['off']
        groups = {ef: loff[lcidx == ci] for ci, ef in cidxmap.items()}

        out = []
        for (etype, fidx), offs in groups.items():
            if etype not in smesh.eidxs:
                continue

            # Keep only those periodic elements that live on this rank
            eidxs = smesh.eidxs[etype]
            eoff = search_unsorted(eidxs, offs[np.isin(offs, eidxs)])
            if len(eoff):
                out.append((etype, fidx, eoff))

        return out


class SpanwiseSnapshotRegion(BaseSnapshotRegion):
    def __init__(self, mesh, cfg, *, source, nstations=None, refpts_fn=None,
                 clean=True):
        super().__init__(mesh, cfg)

        self._source = source
        self._nstations = nstations
        self._refpts_fn = refpts_fn
        self.clean = clean

        self._averager = None
        self.etypes = []
        self.ploc = {}

    def _configure_form(self, snap):
        pris_names = list(snap.pris_names)
        if snap.has_grads:
            self._field_names = pris_names + [f'{f}-{d}' for f in pris_names
                                              for d in range(self.ndims)]
        else:
            self._field_names = pris_names
        self._build_geometry()

    def _build_geometry(self):
        self.axis = self._derive_axis()

        order = self.cfg.getint('solver', 'order')
        if self._refpts_fn is not None:
            self._vis_pts_2d = {
                it: self._refpts_fn(subclass_where(BaseShape, name=it))
                for it in ('quad', 'tri')
            }
        else:
            self._vis_pts_2d = {
                it: subclass_where(BaseShape, name=it).std_ele(order)
                for it in ('quad', 'tri')
            }
        self._vis_div = order

        shapes = {et: subclass_where(BaseShape, name=et)(
                        self.mesh.spts[et].shape[0], self.cfg)
                  for et in self.mesh.spts}

        kwargs = dict(mesh=self.mesh, cfg=self.cfg, shapes=shapes,
                      axis=self.axis, nfields=len(self._field_names),
                      vis_pts_2d=self._vis_pts_2d)

        kind, name, *_ = self._source

        # Hex/pri meshes get the more efficient extruded path when valid.
        self._averager = None
        if {'hex', 'pri'}.issuperset(self.mesh.etypes):
            try:
                self._averager = ExtrudedSpanwise(**kwargs)
            except ValueError:
                pass

        # Otherwise, fall back to sampling
        if self._averager is None:
            eles = self._eles_for_sampled()
            ns = self._nstations or 4*(self.cfg.getint('solver', 'order')
                                       + 1)
            self._averager = SampledSpanwise(
                **kwargs, eles=eles, nstations=ns, **{kind: name})

        self._init_topology()

    def _eles_for_sampled(self):
        meshf = getattr(self.mesh, 'fname', None)
        if meshf is None:
            raise RuntimeError(
                'No spanwise averager available: ExtrudedSpanwise rejected '
                'the mesh (not pure hex/pri, or not axis-flat) and '
                'SampledSpanwise needs a reader-backed mesh (mesh.fname is '
                'not set on in-situ meshes).')

        reader = NativeReader(meshf, None, construct_con=True)
        return reader.eles

    def _init_topology(self):
        shapes = self._averager.shape
        zero = {et: np.zeros((len(self._field_names),
                              shapes[et].nupts,
                              self.mesh.spts[et].shape[1]))
                for et in self.mesh.spts if et in shapes}

        records = self._averager.average(zero)

        grouped = defaultdict(list)
        for itype, vsoln, vpts, cnodes in records:
            grouped[itype].append((vsoln, vpts, cnodes))

        self.etypes = sorted(grouped)
        self._owned = dict(grouped)

        self._cnodes = {it: np.concatenate([c for _, _, c in g])
                        for it, g in self._owned.items()}
        self._ncells = {it: self._cnodes[it].shape[0] for it in self.etypes}

        if self.clean and self.etypes:
            shared = np.fromiter(self.mesh.shared_nodes.by_node, dtype=int)
            self.cleaner = CleanToGrid(
                self._cnodes,
                {it: self._vis_div for it in self.etypes},
                {it: self._vis_pts_2d[it] for it in self.etypes},
                shared)
        else:
            self.cleaner = NullCleaner()

        self.ploc = {}
        self._nsvpts_at = {}
        for itype, groups in self._owned.items():
            vpts = np.concatenate([v for _, v, _ in groups]).swapaxes(0, 1)
            vpts[..., self.axis] = 0
            self._nsvpts_at[itype] = vpts.shape[0]
            self.ploc[itype] = np.ascontiguousarray(
                self.cleaner.select(itype, vpts).T)

    def _compute_sample(self, sample, snap):
        self._ensure_form(snap, lambda: self._configure_form(snap))

        # Pre-process the solution for each element type
        soln = {}
        for et, data in snap.data.items():
            if et not in self.mesh.spts:
                continue
            d = data.swapaxes(0, 1)
            pris = snap.to_pris(d)
            stacked = np.array(pris)

            if snap.has_grads:
                g = snap.grad_data[et]
                grad_pris_list = snap.to_grad_pris(d, g.transpose(2, 0, 1, 3))
                gflat = np.concatenate(
                    [gp for gp in grad_pris_list], axis=0)
                stacked = np.concatenate([stacked, gflat], axis=0)

            soln[et] = stacked

        # Perform the averaging and group results by output face type
        grouped = defaultdict(list)
        for itype, vsoln, _vpts, _cn in self._averager.average(soln):
            grouped[itype].append(vsoln)

        npri = len(snap.pris_names)
        nfields_total = len(self._field_names)
        sample.pris = {}
        sample.grad_pris = {}
        for itype in self.etypes:
            groups = grouped.get(itype, [])
            if not groups:
                sample.pris[itype] = [np.empty(0) for _ in range(npri)]
                if snap.has_grads:
                    sample.grad_pris[itype] = [
                        np.empty((self.ndims, 0)) for _ in range(npri)]
                else:
                    sample.grad_pris[itype] = None
                continue

            # (ncols, nsvpts, nfields) → (nsvpts, ncols, nfields)
            raw = np.concatenate(groups).swapaxes(0, 1)
            avg = self.cleaner.average({itype: raw}, nfields_total, self.dtype)
            vsoln = avg[itype].T
            sample.pris[itype] = [vsoln[i] for i in range(npri)]
            if snap.has_grads:
                grads = vsoln[npri:].reshape(npri, self.ndims, vsoln.shape[1])
                sample.grad_pris[itype] = [grads[i] for i in range(npri)]
            else:
                sample.grad_pris[itype] = None

    def cell_curved(self, itype):
        return np.ones(self._ncells[itype], dtype=bool)

    def _derive_axis(self):
        match self._source:
            case ('periodic', name):
                T = self.mesh.raw[f'periodic/{name}'].attrs['T']
                return self._axis_from_vec(T)
            case ('boundary', namelo, namehi):
                clo = self._boundary_centroid(namelo)
                chi = self._boundary_centroid(namehi)
                return self._axis_from_vec(chi - clo)
            case _:
                raise ValueError(f'Unknown spanwise source: {self._source!r}')

    @staticmethod
    def _axis_from_vec(v, tol=0.1):
        v = np.abs(v) / np.linalg.norm(v)
        aidx = int(np.argmax(v))

        if 1 - v[aidx] > tol:
            raise ValueError('Pair is not axis-aligned')

        return aidx

    def _boundary_centroid(self, name):
        comm, _, _ = get_comm_rank_root()
        mesh = self.mesh

        lsum, ln = np.zeros(3), 0

        if (bc := mesh.bcon.get(name)) is not None:
            for etype, _, eidxs in bc.items():
                lsum += mesh.spts[etype][:, eidxs].sum(axis=(0, 1))
                ln += mesh.spts[etype].shape[0]*len(eidxs)

        comm.Allreduce(mpi.IN_PLACE, lsum, op=mpi.SUM)
        ln = scal_coll(comm.Allreduce, ln, op=mpi.SUM)
        return lsum / ln
