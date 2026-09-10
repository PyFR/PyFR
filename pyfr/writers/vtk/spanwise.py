import re
from collections import defaultdict, namedtuple

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import (AlltoallMixin, get_comm_rank_root, home_rank, mpi,
                          scal_coll)
from pyfr.nputil import batched_fuzzysort, range_offsets, search_unsorted
from pyfr.plugins.common import get_elementscls
from pyfr.points import PointLocator
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape, interp_pts, proj_pts
from pyfr.subdiv import get_subdiv
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter


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


_EInfo = namedtuple('_EInfo', 'spanop geop smpop cnidx')


class ExtrudedSpanwise(_SpanwiseBase, AlltoallMixin):
    def __init__(self, mesh, cfg, shapes, *, nfields, vis_pts_2d, axis=2):
        super().__init__(mesh, cfg, shapes, axis=axis, vis_pts_2d=vis_pts_2d)

        if set(self.pmesh.etypes) - {'hex', 'pri'}:
            raise ValueError(f'Extruded spanwise needs a hex/pri mesh')

        self.nfields = nfields

        # Geometry-only element containers for the physical face normals
        ecls = get_elementscls(cfg)
        eles = {et: ecls(subclass_where(BaseShape, name=et), spts, cfg)
                for et, spts in self.pmesh.spts.items()}

        # Classify the faces of each element by their physical normals
        self.ends = {et: self._classify_faces(et, e) for et, e in eles.items()}
        bad = any(bad.any() for _, _, bad in self.ends.values())

        # The extrusion verdict must be collective for a clean fallback
        comm, _, _ = get_comm_rank_root()
        if scal_coll(comm.Allreduce, int(bad), op=mpi.LOR):
            raise ValueError('Mesh is not extruded along the span axis')

        # Column record dtypes for each element type
        self.cdtype = {et: self._cdtype(et) for et in self.pmesh.etypes}

        # Pre-compute the global span length (single Allreduce)
        slo, shi = self.span_extent()
        self.span_len = shi - slo

        # Pre-compute per-element column ids
        self.col_ids = self._compute_col_ids()

    def _classify_faces(self, et, eles, tol=1e-6):
        b = self.shape[et]
        neles = self.pmesh.spts[et].shape[1]

        bot, top = np.full(neles, -1), np.full(neles, -1)
        bad = np.zeros(neles, dtype=bool)

        for fidx, (*_, norm) in enumerate(b.faces):
            # Unit physical normals at the flux points of this face
            fpts = b.fpts[b.facefpts[fidx]]
            pn = eles.pnorm_at(fpts, np.tile(norm, (len(fpts), 1)))
            d = pn[..., self.axis] / np.linalg.norm(pn, axis=-1)

            # A face is an end or a lateral wall at every flux point
            botc = (d < tol - 1).all(axis=0)
            topc = (d > 1 - tol).all(axis=0)
            lat = (np.abs(d) < tol).all(axis=0)

            bad |= ~(botc | topc | lat)
            bad |= (botc & (bot != -1)) | (topc & (top != -1))
            bot[botc], top[topc] = fidx, fidx

        # Extruded elements have exactly one bottom/top face pair
        bad |= (bot == -1) | (top == -1)

        return bot, top, bad

    def _cdtype(self, et):
        itype = 'quad' if et == 'hex' else 'tri'
        ishapecls = subclass_where(BaseShape, name=itype)

        n2ds = len(self._sample_pts_2d(itype))
        nspts2d = len(ishapecls.std_ele(self.shape[et].nsptsord))

        return np.dtype([
            ('key', np.int64),
            ('sum', np.float64, (n2ds, self.nfields)),
            ('xy', np.float64, (nspts2d, 2)),
            ('cnodes', np.int64, (ishapecls.npts_from_order(1),)),
            ('vperm', np.int32, (n2ds,))
        ])

    @memoize
    def _face_ops(self, et, fidx):
        shape = self.shape[et]
        kind, proj, norm = shape.faces[fidx]
        ishapecls = subclass_where(BaseShape, name=kind)

        spts2d = ishapecls.std_ele(shape.nsptsord)
        pts2d = self._sample_pts_2d(kind)
        pts1d = np.linspace(-1, 1, self.order + 1)

        # Extrude the bottom face lattice through the element
        f2d = proj_pts(proj, pts2d)
        dvec = -np.asarray(norm) / np.linalg.norm(norm)
        eq3d = np.vstack([f2d + (1 + t)*dvec for t in pts1d])

        # Fused interpolation + span-averaging operator
        eqop = shape.ubasis.nodal_basis_at(eq3d)
        swts = 0.5*2**0.5*get_polybasis('line', self.order, pts1d).invvdm[:, 0]
        spanop = (swts @ eqop.reshape(len(pts1d), -1)).reshape(len(pts2d), -1)

        # Footprint geometry and sample point placement operators
        geop = shape.sbasis.nodal_basis_at(proj_pts(proj, spts2d))
        smpop = shape.sbasis.nodal_basis_at(f2d)

        # Corner shape points of the face for the output topology
        cnidx = shape.face_corner_pts_idxs(fidx, shape.nspts)

        return _EInfo(spanop, geop, smpop, cnidx)

    @memoize
    def _shape_op(self, et):
        itype = 'quad' if et == 'hex' else 'tri'
        sptord = self.shape[et].nsptsord

        ishapecls = subclass_where(BaseShape, name=itype)
        spts2d = ishapecls.std_ele(sptord)

        basis = get_polybasis(itype, sptord, spts2d)
        return basis.nodal_basis_at(self.vis_pts_2d[itype])

    def average(self, soln):
        out = []
        for et in self.cdtype:
            cols = self._reduce_etype(et, soln.get(et))
            if len(cols):
                itype = 'quad' if et == 'hex' else 'tri'

                # Restore the 2D reference ordering of each column
                means = np.empty_like(cols['sum'])
                np.put_along_axis(means, cols['vperm'][..., None],
                                  cols['sum'] / self.span_len, axis=1)

                vsoln = self._vis_op(itype) @ means
                vxy = self._shape_op(et) @ cols['xy']
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

        # Positions of each subset element within the parent mesh
        pos = {et: search_unsorted(self.pmesh.eidxs[et], e)
               for et, e in self.mesh.eidxs.items()}

        # Bottom faces of each subset element for span integration
        self.bot_faces = {et: self.ends[et][0][p] for et, p in pos.items()}

        # Slice the full-mesh ids by each subset etype's eidx positions
        return {et: col[offs[et] + p] for et, p in pos.items()}

    def _end_mask(self, con):
        # Interface instances whose face is a column end
        m = np.zeros(len(con), dtype=bool)
        for etype, fidx, eidxs, rows in con.foreach():
            bot, top, _ = self.ends[etype]
            m[rows] = (bot[eidxs] == fidx) | (top[eidxs] == fidx)

        return m

    def _span_links(self, offs):
        # Per-cidx element type offset table
        lhs, rhs = self.pmesh.con
        cmap = lhs.cidxmap
        cidxoff = np.zeros(max(cmap) + 1, dtype=int)
        for cidx, (etype, _) in cmap.items():
            if etype in offs:
                cidxoff[cidx] = offs[etype]

        # Local internal pairs across spanwise faces
        m = self._end_mask(lhs) | self._end_mask(rhs)
        pa = cidxoff[lhs.cidxs[m]] + lhs.eidxs[m]
        pb = cidxoff[rhs.cidxs[m]] + rhs.eidxs[m]

        # Inter-rank spanwise face pairs
        links = [(nr, cidxoff[c.cidxs] + c.eidxs, mk)
                 for nr, c in self.pmesh.con_p.items()
                 if (mk := self._end_mask(c)).any()]

        return pa, pb, links

    def _reduce_etype(self, etype, soln_pre):
        cdtype = self.cdtype[etype]

        if (spts := self.mesh.spts.get(etype)) is None:
            return self._exchange(np.empty(0, dtype=cdtype))

        # Column ids, bottom faces and per-element span lengths
        keys = self.col_ids[etype]
        bot = self.bot_faces[etype]
        ds = np.ptp(spts[..., self.axis], axis=0)
        snodes = self.mesh.spts_nodes[etype]

        eles = np.empty(spts.shape[1], dtype=cdtype)
        eles['key'] = keys

        for fidx in np.unique(bot):
            ei = self._face_ops(etype, int(fidx))
            sel = np.flatnonzero(bot == fidx)

            # Interpolate to the bottom face lattice and span-integrate
            v = (ei.spanop @ soln_pre[..., sel]).transpose(2, 1, 0)

            # Canonically order the lattice by its physical position
            sp = interp_pts(ei.smpop, spts[:, sel])
            perm = batched_fuzzysort(sp.transpose(1, 2, 0)[:, self.in_axes])

            eles['sum'][sel] = np.take_along_axis(v, perm[..., None], axis=1)
            eles['vperm'][sel] = perm

            # Footprint geometry and output topology
            fp = interp_pts(ei.geop, spts[:, sel])
            eles['xy'][sel] = fp.transpose(1, 0, 2)[..., self.in_axes]
            eles['cnodes'][sel] = snodes[sel][:, ei.cnidx]

        # Weight the per-element sums by their span lengths
        eles['sum'] *= ds[:, None, None]

        return self._exchange(self._merge_cols(eles))

    @staticmethod
    def _merge_cols(cols):
        # Merge columns with a common key, accumulating their sums
        ukeys, first, inverse = np.unique(cols['key'], return_index=True,
                                          return_inverse=True)
        out = cols[first]
        out['sum'] = 0
        np.add.at(out['sum'], inverse, cols['sum'])

        return out

    def _exchange(self, cols):
        comm, _, _ = get_comm_rank_root()

        # Route each column to its home rank by hashed key
        home = home_rank(cols['key'], comm.size)
        hcount = np.bincount(home, minlength=comm.size)
        resp = self._alltoallcv(comm, cols[np.argsort(home)], hcount)[0]

        # Multiple senders may have contributed to the same column
        return self._merge_cols(resp)


class SampledSpanwise(_SpanwiseBase, AlltoallMixin):
    def __init__(self, mesh, cfg, shapes, *, nfields, nstations,
                 vis_pts_2d, axis=2, boundary=None, periodic=None):
        super().__init__(mesh, cfg, shapes, axis=axis, vis_pts_2d=vis_pts_2d)

        if int(nstations) < 2:
            raise ValueError('nstations must be >= 2 (trapezium rule)')

        self.boundary = boundary
        self.periodic = periodic
        self.nstations = int(nstations)
        self.nfields = nfields

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

        vpts = interp_pts(mesh_op, self.mesh.spts[etype][:, idxs])
        spts = interp_pts(smesh_op, self.mesh.spts[etype][:, idxs])
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
            vsoln = interp_pts(self._vis_op(itype), m.swapaxes(0, 1))
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

        # Unknown boundaries fail globally via the codec
        if f'bc/{name}' not in smesh.codec:
            raise ValueError(f'Unknown boundary {name!r}')

        # Subset solutions must retain every local boundary face
        pcon, con = rmesh.bcon.get(name), smesh.bcon.get(name)
        if smesh.subset and len(con or []) != len(pcon or []):
            raise ValueError('Boundary not present in subset solution')

        return list(con.items()) if con is not None else []

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


class VTKSpanwiseWriter(BaseVTKWriter):
    type = 'spanwise'
    adapter_kind = 'volume'
    output_curved = True
    needs_con = True
    dimensions = '3'

    def __init__(self, meshf, *, nstations=None, boundary=None, periodic=None,
                 **kwargs):
        super().__init__(meshf, **kwargs)

        if boundary is None and periodic is None:
            raise ValueError('Boundary or periodic must be specified')
        if boundary is not None and periodic is not None:
            raise ValueError('Boundary and periodic are mutually exclusive')

        self.nstations = int(nstations) if nstations else None

        if periodic is not None:
            self._source = ('periodic', periodic)
        else:
            parts = boundary.split(',')
            if len(parts) != 2:
                raise ValueError('Boundary must be a comma separated pair')

            self._source = ('boundary', parts[0].strip(), parts[1].strip())

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        self.order = self.cfg.getint('solver', 'order')
        self.axis = self._derive_axis()

        # Construct the averager
        savg = self._get_averager()

        # Pre-process the solution for each element type
        soln = {et: self._soln_pre(et) for et in self.soln.data}

        # Perform the averaging and group results by output face type
        self._owned = owned = defaultdict(list)
        for itype, vsoln, vpts, cnodes in savg.average(soln):
            owned[itype].append((vsoln, vpts, cnodes))

        # Output element types we own non-trivial data for
        self.einfo = [(itype, sum(len(m) for m, *_ in groups))
                      for itype, groups in owned.items()]

    def _build_extra_fields(self):
        # Derived fields come from span-averaged means: average then derive
        self._extra_fields = {}
        self._register_pp_fields()

    def _prepare_pts(self, itype):
        groups = self._owned[itype]

        means = np.concatenate([m for m, _, _ in groups]).transpose(1, 2, 0)
        vpts = np.concatenate([v for _, v, _ in groups]).swapaxes(0, 1)
        vpts[..., self.axis] = 0

        # Derive quantities from the span-averaged means at the 2D footprint
        pointf = self._postproc(means.swapaxes(0, 1), vpts.transpose(2, 0, 1))

        return vpts, means, np.ones(vpts.shape[1], dtype=bool), {}, pointf

    def _output_topology(self):
        cnodes, svpts = {}, {}
        for itype, groups in self._owned.items():
            cnodes[itype] = np.concatenate([c for _, _, c in groups])
            svpts[itype] = self._svpts2d(itype)

        return cnodes, svpts

    def _soln_pre(self, etype):
        return self._pre_proc_fields(self.soln.data[etype].swapaxes(0, 1))


    @memoize
    def _svpts2d(self, itype):
        ishapecls = subclass_where(BaseShape, name=itype)
        svpts = ishapecls.std_ele(self.etypes_div[itype])
        if self.ho_output:
            sdiv = get_subdiv(itype, self.etypes_div[itype])
            svpts = svpts[sdiv.vtk_nodemaps[len(svpts)]]

        return svpts

    def _vis_pts_2d(self):
        return {it: self._svpts2d(it) for it in ('quad', 'tri')}

    def _derive_axis(self):
        match self._source:
            case ('periodic', name):
                if (p := self.mesh.raw.get(f'periodic/{name}')) is None:
                    raise ValueError(f'Unknown periodic boundary {name!r}')

                return self._axis_from_vec(p.attrs['T'])
            case ('boundary', namelo, namehi):
                clo = self._boundary_centroid(namelo)
                chi = self._boundary_centroid(namehi)
                return self._axis_from_vec(chi - clo)

    @staticmethod
    def _axis_from_vec(v, tol=0.1):
        v = np.abs(v) / np.linalg.norm(v)
        aidx = int(np.argmax(v))

        if 1 - v[aidx] > tol:
            raise ValueError('Pair is not axis-aligned')

        return aidx

    def _boundary_centroid(self, name):
        comm, _, _ = get_comm_rank_root()
        mesh = self.reader.mesh

        if f'bc/{name}' not in mesh.codec:
            raise ValueError(f'Unknown boundary {name!r}')

        lsum, ln = np.zeros(3), 0

        # If we have faces on this boundary then sum their shape points
        if (bc := mesh.bcon.get(name)) is not None:
            for etype, _, eidxs in bc.items():
                lsum += mesh.spts[etype][:, eidxs].sum(axis=(0, 1))
                ln += mesh.spts[etype].shape[0]*len(eidxs)

        # Reduce to compute the global centroid
        comm.Allreduce(mpi.IN_PLACE, lsum, op=mpi.SUM)
        ln = scal_coll(comm.Allreduce, ln, op=mpi.SUM)
        return lsum / ln

    def _get_averager(self):
        kind, name, *_ = self._source
        shapes = {et: self._get_shape(et, self.cfg)
                  for et in self.reader.mesh.etypes}

        nfields = len(self.soln.layout or self._soln_fields)
        kwargs = dict(mesh=self.mesh, cfg=self.cfg, shapes=shapes,
                      axis=self.axis, nfields=nfields,
                      vis_pts_2d=self._vis_pts_2d())

        # If the mesh is hex/pri then it may be extruded; assume as
        # such and try to use the more efficient extruded path
        if {'hex', 'pri'}.issuperset(self.reader.mesh.etypes):
            try:
                return ExtrudedSpanwise(**kwargs)
            except ValueError:
                pass

        # Otherwise, fall back to sampling
        nstations = self.nstations or 4*(self.order + 1)
        return SampledSpanwise(**kwargs, nstations=nstations, **{kind: name})
