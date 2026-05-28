from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.postproc.adapters import (BoundaryPostProcData,
                                            VolumePostProcData)
from pyfr.plugins.soln.insitu.base import build_cleaner
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers.vtk.shapes import get_vtk_shape


class VolumeOutput:
    kind = 'volume'

    def __init__(self, renderer, sname='volume', clean=False):
        self.renderer = renderer
        self.sname = sname
        self.rdata = renderer.rdata
        self.soln_ops = {}
        self._setup_clean(clean)

    def _setup_clean(self, clean):
        if clean:
            self.cleaner = self._make_cleaner()
            self.publish_fields = self._publish_clean
        else:
            self.cleaner = None
            self.publish_fields = self._publish_direct

    def _make_cleaner(self):
        mesh, divisor = self.renderer.mesh, self.renderer.divisor
        cnodemap, svptsmap = {}, {}
        for etype, eidxs in self.rdata.items():
            shapecls = subclass_where(BaseShape, name=etype)
            spts_nodes = mesh.spts_nodes[etype]
            cidxs = shapecls.corner_pts_idxs(spts_nodes.shape[1])
            cnodemap[etype] = spts_nodes[eidxs][:, cidxs]
            svptsmap[etype] = shapecls.std_ele(divisor)

        return build_cleaner(mesh, divisor, cnodemap, svptsmap)

    def domain_keys(self):
        return list(self.rdata)

    def _points(self, key, xd):
        if self.cleaner is None:
            return xd.swapaxes(0, 1).reshape(-1, xd.shape[-1]).T
        else:
            return self.cleaner.select(key, xd).T

    def _connectivity(self, key, snodes, neles, nsvpts):
        if self.cleaner is None:
            con = np.tile(snodes, (neles, 1))
            con += (np.arange(neles)*nsvpts)[:, None]
            return con
        else:
            return self.cleaner.layouts[key][0][:, snodes]

    def build_blueprint(self, renderer, domid, etype):
        dom = renderer._domain_path(self.sname, domid)

        soln_op, xd = renderer.adapter.soln_op_vpts(etype, renderer.divisor)
        xd = xd[..., self.rdata[etype]].transpose(0, 2, 1)
        nsvpts, neles, _ = xd.shape

        self.soln_ops[etype] = soln_op

        snodes = get_vtk_shape(etype, renderer.divisor).subnodes
        conn = self._connectivity(etype, snodes, neles, nsvpts)

        renderer._write_domain(dom, self.sname, domid, etype, neles,
                               self._points(etype, xd), conn)
        return dom

    def csolns_cgrads(self, soln, grad_soln, etype):
        soln_op = self.soln_ops[etype]
        rgn = self.rdata[etype]

        # Subset and transpose the solution, then interpolate to subdiv points
        csolns = soln_op @ soln[etype][..., rgn].swapaxes(0, 1)
        if grad_soln is not None:
            # Interpolate the gradients to the points
            cg = np.rollaxis(grad_soln[etype], 2)[..., rgn]
            return csolns, soln_op @ cg
        else:
            return csolns, None

    def run_postproc(self, runner, etype, psolns, pgrads):
        adapter = VolumePostProcData(self.renderer.scfg, psolns, pgrads)
        return runner.run(adapter, public_only=True)

    def _emit(self, mesh_n, dom, fname, arr):
        # Delegates to the renderer so hosts (eg. Catalyst) can override
        # to AoS or other layouts without subclassing the output.
        self.renderer._emit_field(mesh_n, dom, fname, arr)

    def _publish_direct(self, mesh_n, fields):
        for key, items in fields.items():
            dom = self.renderer.dinfo[self.sname, key]
            for fname, arr in items:
                self._emit(mesh_n, dom, fname, arr)

    def _publish_clean(self, mesh_n, fields):
        comm, _, _ = get_comm_rank_root()
        dtype = self.renderer.dtype

        # Re-key by published (possibly namespaced) field name
        byfield = defaultdict(dict)
        for key, items in fields.items():
            for fname, arr in items:
                byfield[fname][key] = arr

        # Sorted per-source list so idle ranks hit every collective
        for field in sorted(self.renderer._source_fields[self.sname]):
            fname = self.renderer._field_name(self.sname, field)
            arrs = byfield.get(fname, {})

            ncomp = max((a.shape[-1] for a in arrs.values()), default=1)
            ncomp = comm.allreduce(ncomp, op=mpi.MAX)

            efields = {key: arr.astype(dtype, copy=False)
                       for key, arr in arrs.items()}

            avg = self.cleaner.average(efields, ncomp, dtype)
            for key, vals in avg.items():
                dom = self.renderer.dinfo[self.sname, key]
                self._emit(mesh_n, dom, fname, vals)


class BoundaryOutput(VolumeOutput):
    kind = 'boundary'

    def __init__(self, renderer, sname, region, clean=False):
        self.renderer = renderer
        self.sname = sname

        # Accept bc/foo or foo to match mesh.bcon keys
        bcname = region.removeprefix('bc/')
        comm, _, _ = get_comm_rank_root()
        if not comm.allreduce(bcname in renderer.mesh.bcon, op=mpi.LOR):
            raise ValueError(f'Boundary {bcname} does not exist')

        # Per-itype patches, each a flat (eidxs, etype, mop, sop, fidx, svpts)
        self.patches = patches = defaultdict(list)
        if (conn := renderer.mesh.bcon.get(bcname)) is not None:
            for etype, fidx, eidxs in conn.items():
                itype, mop, sop, svpts = renderer.adapter.face_soln_op_vpts(
                    etype, fidx, renderer.divisor
                )
                patches[itype].append((eidxs, etype, mop, sop, fidx, svpts))

        self._setup_clean(clean)

    def _make_cleaner(self):
        mesh, divisor = self.renderer.mesh, self.renderer.divisor
        cnodemap, svptsmap = {}, {}
        for itype, patches in self.patches.items():
            ishapecls = subclass_where(BaseShape, name=itype)
            pieces = []
            for eidxs, etype, _, _, fidx, _ in patches:
                shapecls = subclass_where(BaseShape, name=etype)
                spts_nodes = mesh.spts_nodes[etype]
                cidxs = shapecls.face_corner_pts_idxs(fidx,
                                                      spts_nodes.shape[1])
                pieces.append(spts_nodes[np.ix_(eidxs, cidxs)])
            cnodemap[itype] = np.concatenate(pieces)
            svptsmap[itype] = ishapecls.std_ele(divisor)

        return build_cleaner(mesh, divisor, cnodemap, svptsmap)

    def domain_keys(self):
        return list(self.patches)

    def build_blueprint(self, renderer, domid, itype):
        dom = renderer._domain_path(self.sname, domid)
        spts = renderer.mesh.spts

        # Per-patch face coords concatenated to (nsvpts, neles, ndim)
        xparts = []
        for eidxs, etype, mop, _, _, _ in self.patches[itype]:
            sp = spts[etype][:, eidxs]
            vp = (mop @ sp.reshape(len(sp), -1)).reshape(-1, *sp.shape[1:])
            xparts.append(vp)

        xd = np.concatenate(xparts, axis=1)
        nsvpts, neles, _ = xd.shape

        snodes = get_vtk_shape(itype, renderer.divisor).subnodes
        conn = self._connectivity(itype, snodes, neles, nsvpts)

        renderer._write_domain(dom, self.sname, domid, itype, neles,
                               self._points(itype, xd), conn)
        return dom

    def csolns_cgrads(self, soln, grad_soln, itype):
        css, cgs = [], []
        for eidxs, etype, _, sop, _, _ in self.patches[itype]:
            css.append(sop @ soln[etype][..., eidxs].swapaxes(0, 1))
            if grad_soln is not None:
                cg = np.rollaxis(grad_soln[etype], 2)[..., eidxs]
                cgs.append(sop @ cg)

        csolns = np.concatenate(css, axis=2)
        cgrads = np.concatenate(cgs, axis=3) if cgs else None
        return csolns, cgrads

    def run_postproc(self, runner, itype, psolns, pgrads):
        scfg = self.renderer.scfg
        spts = self.renderer.mesh.spts

        # Slice per-patch views, run postprocs, merge by field across patches
        merged = defaultdict(list)
        offset = 0
        for eidxs, etype, _, _, fidx, svpts in self.patches[itype]:
            sl = slice(offset, offset + len(eidxs))
            ppris = [p[..., sl] for p in psolns]
            if pgrads is None:
                ppgrads = None
            else:
                ppgrads = [None if g is None else g[..., sl] for g in pgrads]
            offset = sl.stop

            psp = spts[etype][:, eidxs]
            adapter = BoundaryPostProcData(scfg, ppris, psp, etype, fidx,
                                           svpts, grad_pris=ppgrads)
            for fname, arr in runner.run(adapter, public_only=True).items():
                merged[fname].append(arr)

        return {fname: np.concatenate(parts, axis=1)
                for fname, parts in merged.items()}
