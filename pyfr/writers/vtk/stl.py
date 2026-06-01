from pathlib import Path

import numpy as np

from pyfr.cache import clear_memoize
from pyfr.mpiutil import get_comm_rank_root
from pyfr.polys import TriPolyBasis
from pyfr.shapes import TriShape
from pyfr.snapshot import FileSnapshot
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts
from pyfr.writers.vtk.output import DirectVTKOutput
from pyfr.writers.vtk.shapes import get_vtk_shape


def _vertex_normals_mwa(fnorms, verts, vids):
    # Angle-weighted vertex normals (Thurmer & Wuthrich 1998)
    fnorms = fnorms / np.linalg.norm(fnorms, axis=1, keepdims=True)
    v = verts[vids].swapaxes(0, 1)
    vnorms = np.zeros((len(verts), 3))

    for ci in range(3):
        e1 = v[(ci + 1) % 3] - v[ci]
        e2 = v[(ci + 2) % 3] - v[ci]
        e1n = np.linalg.norm(e1, axis=1)
        e2n = np.linalg.norm(e2, axis=1)
        cos_a = np.clip(np.sum(e1*e2, axis=1) / (e1n*e2n), -1, 1)
        np.add.at(vnorms, vids[:, ci], fnorms*np.arccos(cos_a)[:, None])

    vnorms /= np.linalg.norm(vnorms, axis=1, keepdims=True)
    return vnorms


def _spherigon_smooth(flat_pts, bary, tri_verts, tri_norms):
    # C1 spherigon (Volino & Magnenat-Thalmann 1998)
    ntri = flat_pts.shape[1]
    smoothed = np.empty_like(flat_pts)
    tol = 1e-8

    for s, P, r in zip(smoothed, flat_pts, bary):
        N = np.einsum('k,knj->nj', r, tri_norms)
        N /= np.linalg.norm(N, axis=1, keepdims=True)

        Q, Qp = np.empty((2, 3, ntri, 3))
        for k, (vk, nk) in enumerate(zip(tri_verts, tri_norms)):
            diff = vk - P
            ndiff = np.sum(diff*N, axis=1, keepdims=True)
            K = P + N*ndiff
            denom = 1.0 + np.sum(N*nk, axis=1, keepdims=True)
            t = np.sum((vk - K)*nk, axis=1, keepdims=True) / denom
            Q[k] = K + N*t
            Qp[k] = vk - N*ndiff

        dist2 = np.sum((Qp - P)**2, axis=2)
        blend = np.zeros((3, ntri))

        for k in range(3):
            kp, km = (k + 1) % 3, (k + 2) % 3
            if not (tol < r[k] < 1.0 - tol):
                continue
            wm = dist2[km] / (dist2[km] + dist2[k])
            wp = dist2[kp] / (dist2[kp] + dist2[k])
            blend[k] = r[k]**2*(r[km]**2*wm + r[kp]**2*wp)

        total = blend.sum(axis=0)
        if total.any():
            blend /= total

        for k in range(3):
            if r[k] >= 1.0 - tol:
                blend[k] = 1.0

        for k in range(3):
            if r[k] > tol:
                continue
            kp, km = (k + 1) % 3, (k + 2) % 3
            w = r[kp]**2 + r[km]**2
            blend[kp] = r[kp]**2 / w
            blend[km] = r[km]**2 / w

        s[:] = (Q*blend[:, :, None]).sum(axis=0)

    return smoothed


class VTKSTLWriter(BaseVTKWriter):
    # STL surface export.  The STL is just a set of physical points (welded
    # subdivision vertices) plus user-supplied triangle connectivity — exactly
    # what snap.at_points(ppts) (PointsSnapshotRegion) consumes.  Sampling
    # runs once at the welded vertices; emission expands per-triangle via the
    # weld inverse `pinv` so each triangle gets its own VTU cell.
    type = 'stl'
    output_curved = False
    dimensions = '3'
    _output_cls = DirectVTKOutput

    def __init__(self, meshf, stlrgns, *, subdiv='linear', **kwargs):
        # STL: never HO, always discontinuous (per-triangle output)
        kwargs['order'] = None
        kwargs['discontinuous'] = True
        divisor = kwargs.setdefault('divisor', 1)

        super().__init__(meshf, **kwargs)

        if subdiv not in ('linear', 'spherigon'):
            raise ValueError(f'Invalid subdiv type: {subdiv}')

        # Read + merge the STL surfaces from the mesh, subdivide, weld unique
        stl = np.vstack([self.reader.mesh.raw[f'regions/stl/{s}']
                         for s in stlrgns])
        pts = self._subdivide_pts(stl, divisor, subdiv)  # (n_subdiv, ntri, 3)
        ppts, pinv = np.unique(pts.reshape(-1, 3), axis=0,
                               return_inverse=True)

        self._stl_pts_shape = pts.shape    # (n_subdiv, ntri, 3)
        self._stl_ppts = ppts              # (n_welded, 3)
        self._stl_pinv = pinv              # (n_subdiv*ntri,)

    def _emit_fields(self, kind):
        # STL emits per-triangle on welded vertices; no per-element cell data
        # exists on the surface.  Filter cell-kind aux out — base's emit loops
        # walk this generator uniformly.
        if kind == 'cell':
            return
        yield from super()._emit_fields(kind)

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)
        _, rank, root = get_comm_rank_root()
        if rank == root:
            self.einfo = [('tri', self._stl_pts_shape[1])]
        else:
            self.einfo = []

    def process(self, solnf, outfname):
        clear_memoize(self)
        self._load_soln(solnf)

        # Sample at welded STL vertices — PointSampler under the hood gathers
        # to root; non-root ranks see empty arrays (their einfo is also empty).
        self._snap = FileSnapshot.from_loaded(self.mesh, self.soln)
        self._region = self._snap.at_points(self._stl_ppts)
        self._sample = self._region.sample(self._snap)

        if self._field_names:
            self._sample.run(self.field_runner, public_only=True)

        # _output_cls.npts is the only piece the base machinery actually uses
        # (for _get_npts_ncells_nnodes_ho); points/connectivity emitted by our
        # _write_data below.
        self._output = self._output_cls(self)

        if Path(outfname).suffix == '.vtu':
            self._write_vtu(outfname)
        else:
            self._write_pvtu(outfname)

    def _point_field_data(self, etype):
        # Sampled values are on welded vertices; expand back per-triangle
        # via pinv so emission matches DirectVTKOutput's element-major layout.
        sample = self._sample
        pinv = self._stl_pinv
        n_subdiv, ntri = self._stl_pts_shape[:2]
        fields = []

        for name, info in sample.fields_meta.items():
            if info.kind != 'point':
                continue
            if name in self._remove_fields:
                continue

            arr = sample.field_array('points', info)
            if arr is None:
                continue

            if info.source in ('primitive', 'gradient'):
                # field_array returns (npts, ncomp); fan out via pinv.
                arr = arr[pinv].reshape(n_subdiv, ntri, arr.shape[-1])
                ftype = self.dtype
            else:
                # Aux / provider: (npts,) or (ncomp, npts) — expand via pinv.
                ftype = info.dtype
                if arr.ndim == 1:
                    arr = arr[pinv].reshape(n_subdiv, ntri, 1)
                else:
                    arr = arr[:, pinv].reshape(arr.shape[0], n_subdiv,
                                               ntri).transpose(1, 2, 0)

            fields.append((np.ascontiguousarray(arr, dtype=ftype), ftype))

        return fields

    def _write_data(self, write, etype):
        # Per-triangle vpts from welded region.ploc + pinv -> standard
        # DirectVTKOutput emit (tile-pattern connectivity).
        region = self._region
        ppts = region.ploc['points']           # (3, n_welded)
        pinv = self._stl_pinv
        n_subdiv, ntri = self._stl_pts_shape[:2]
        nsvpts = n_subdiv

        per_tri_vpts = ppts.T[pinv].reshape(n_subdiv, ntri, 3)

        out = self._output.points(etype, per_tri_vpts)
        self._write_darray(out, write, self.dtype)

        if self.ho_output:
            nodes = np.arange(nsvpts)
            subcellsoff = nsvpts
            types = get_vtk_shape(etype, self.etypes_div[etype]).vtk_ho_type
        else:
            subdiv = get_vtk_shape(etype, self.etypes_div[etype])
            nodes = subdiv.subnodes
            subcellsoff = subdiv.subcelloffs
            types = subdiv.subcelltypes

        vtu_con = self._output.connectivity(etype, nodes, ntri, nsvpts)
        vtu_off = np.tile(subcellsoff, (ntri, 1))
        vtu_off += (np.arange(ntri)*len(nodes))[:, None]
        vtu_typ = np.tile(types, ntri)

        self._write_darray(vtu_con, write, np.int64)
        self._write_darray(vtu_off, write, np.int64)
        self._write_darray(vtu_typ, write, np.uint8)

        for arr, dtype in self._output.point_fields(etype):
            self._write_darray(arr, write, dtype)

    def _subdivide_pts(self, stl, order, subdiv):
        basis = TriPolyBasis(1, TriShape.std_ele(1))
        op = basis.nodal_basis_at(TriShape.std_ele(order))

        pts = interpolate_pts(op, stl[:, 1:].swapaxes(0, 1))

        if subdiv == 'spherigon' and order > 1:
            fnorms = stl[:, 0].astype(float)
            verts = stl[:, 1:].reshape(-1, 3)

            uverts, vids = np.unique(verts, axis=0, return_inverse=True)
            vids = vids.reshape(-1, 3)

            vnorms = _vertex_normals_mwa(fnorms, uverts, vids)

            spts = TriShape.std_ele(order)
            bary = np.column_stack([-(spts[:, 0] + spts[:, 1]) / 2,
                                    (1 + spts[:, 0]) / 2,
                                    (1 + spts[:, 1]) / 2])

            tri_v = uverts[vids].swapaxes(0, 1)
            tri_n = vnorms[vids].swapaxes(0, 1)

            pts = _spherigon_smooth(pts, bary, tri_v, tri_n)

        return pts
