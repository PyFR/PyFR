import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.polys import TriPolyBasis
from pyfr.shapes import TriShape
from pyfr.snapshot.region import PointsSnapshotRegion, interp_ops
from pyfr.writers.vtk.base import BaseVTKWriter
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

        # Accumulate angle-weighted face normal onto each vertex
        np.add.at(vnorms, vids[:, ci], fnorms*np.arccos(cos_a)[:, None])

    vnorms /= np.linalg.norm(vnorms, axis=1, keepdims=True)
    return vnorms


def _spherigon_smooth(flat_pts, bary, tri_verts, tri_norms):
    # C1 spherigon (Volino & Magnenat-Thalmann 1998); lifts flat
    # subdivision points onto curved tangent planes defined by
    # per-vertex normals
    ntri = flat_pts.shape[1]
    smoothed = np.empty_like(flat_pts)
    tol = 1e-8

    # Iterate over subdivision points (same bary coords for all tris)
    for s, P, r in zip(smoothed, flat_pts, bary):
        # Phong-interpolated normal at P
        N = np.einsum('k,knj->nj', r, tri_norms)
        N /= np.linalg.norm(N, axis=1, keepdims=True)

        # Per-vertex C1 target points Q and projections Qp (eqs 2, 3, 8)
        Q, Qp = np.empty((2, 3, ntri, 3))
        for k, (vk, nk) in enumerate(zip(tri_verts, tri_norms)):
            diff = vk - P
            ndiff = np.sum(diff*N, axis=1, keepdims=True)
            K = P + N*ndiff
            denom = 1.0 + np.sum(N*nk, axis=1, keepdims=True)
            t = np.sum((vk - K)*nk, axis=1, keepdims=True) / denom
            Q[k] = K + N*t
            Qp[k] = vk - N*ndiff

        # C1 SuperBlend weights (eq 10)
        dist2 = np.sum((Qp - P)**2, axis=2)
        blend = np.zeros((3, ntri))

        for k in range(3):
            kp, km = (k + 1) % 3, (k + 2) % 3
            if not (tol < r[k] < 1.0 - tol):
                continue
            wm = dist2[km] / (dist2[km] + dist2[k])
            wp = dist2[kp] / (dist2[kp] + dist2[k])
            blend[k] = r[k]**2*(r[km]**2*wm + r[kp]**2*wp)

        # Normalise blend weights
        total = blend.sum(axis=0)
        if total.any():
            blend /= total

        # At a vertex the SuperBlend is degenerate; use Q[k] directly
        for k in range(3):
            if r[k] >= 1.0 - tol:
                blend[k] = 1.0

        # On an edge the C1 blend is ill-conditioned; fall back to C0
        for k in range(3):
            if r[k] > tol:
                continue
            kp, km = (k + 1) % 3, (k + 2) % 3
            w = r[kp]**2 + r[km]**2
            blend[kp] = r[kp]**2 / w
            blend[km] = r[km]**2 / w

        # Blend
        s[:] = (Q*blend[:, :, None]).sum(axis=0)

    return smoothed


class VTKSTLWriter(BaseVTKWriter):
    # STL surface export.  The STL is just a set of physical points (welded
    # subdivision vertices) plus user-supplied triangle connectivity — exactly
    # what snap.at_points(ppts) (PointsSnapshotRegion) consumes.  Sampling
    # runs once at the welded vertices; emission expands per-triangle via the
    # weld inverse `pinv` so each triangle gets its own VTU cell.
    #
    # STL keeps its own _write_data because per-triangle emission needs the
    # weld-inverse expansion, which doesn't fit the region.connectivity API
    # (region holds welded points, no per-element layout).  DirectVTKOutput
    # provides the points/connectivity adapter for that path.
    type = 'stl'
    output_curved = False
    dimensions = '3'
    pyr_divisor_bump = 2

    def __init__(self, mesh, cfg, stlrgns, *, subdiv='linear', **kwargs):
        if not stlrgns:
            raise ValueError('STL export requires at least one region')

        # STL: never HO, always discontinuous (per-triangle output).
        kwargs['order'] = None
        kwargs['discontinuous'] = True
        if kwargs.get('divisor') is None:
            kwargs['divisor'] = 1

        if subdiv not in ('linear', 'spherigon'):
            raise ValueError(f'Invalid subdiv type: {subdiv}')

        self._stlrgns = stlrgns
        self._subdiv = subdiv
        self._init_divisor = kwargs['divisor']

        super().__init__(mesh, cfg, **kwargs)

        self._output = DirectVTKOutput(self)

    def _init_einfo(self):
        # Read and merge the STL surfaces from the mesh
        stl = np.vstack([self.mesh.raw[f'regions/stl/{s}']
                         for s in self._stlrgns])

        # Subdivide the mesh
        pts = self._subdivide_pts(stl, self._init_divisor, self._subdiv)

        # Weld coincident vertices
        ppts, pinv = np.unique(pts.reshape(-1, 3), axis=0,
                               return_inverse=True)

        self._stl_pts_shape = pts.shape    # (n_subdiv, ntri, 3)
        self._stl_ppts = ppts              # (n_welded, 3)
        self._stl_pinv = pinv              # (n_subdiv*ntri,)

        _, rank, root = get_comm_rank_root()
        if rank == root:
            self.einfo = [('tri', self._stl_pts_shape[1])]
        else:
            self.einfo = []

    def _build_region(self):
        return PointsSnapshotRegion(self.mesh, self.cfg, self._stl_ppts)

    def _get_npts_ncells_nnodes_lin(self, etype, neles):
        nsvpts = self._nsvpts(etype)
        return neles*nsvpts, neles, neles*nsvpts

    def _emit_fields(self, kind):
        # STL carries no per-element cell data; keep only point fields
        if kind == 'cell':
            return
        yield from super()._emit_fields(kind)

    def _point_field_data(self, etype):
        sample = self._sample
        pinv = self._stl_pinv
        n_subdiv, ntri = self._stl_pts_shape[:2]
        fields = []

        for name, info in sample.fields.items():
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
                    ncomp = arr.shape[0]
                    arr = arr[:, pinv].reshape(ncomp, n_subdiv, ntri)
                    arr = arr.transpose(1, 2, 0)

            fields.append((arr, ftype))

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
        # Flat linear subdivision
        basis = TriPolyBasis(1, TriShape.std_ele(1))
        op = basis.nodal_basis_at(TriShape.std_ele(order))

        pts = interp_ops(op, stl[:, 1:].swapaxes(0, 1), cast=True)

        if subdiv == 'spherigon' and order > 1:
            fnorms = stl[:, 0].astype(float)
            verts = stl[:, 1:].reshape(-1, 3)

            # Weld coincident vertices
            uverts, vids = np.unique(verts, axis=0, return_inverse=True)
            vids = vids.reshape(-1, 3)

            # Angle-weighted vertex normals
            vnorms = _vertex_normals_mwa(fnorms, uverts, vids)

            # Barycentric coordinates at the subdivision points
            spts = TriShape.std_ele(order)
            bary = np.column_stack([-(spts[:, 0] + spts[:, 1]) / 2,
                                    (1 + spts[:, 0]) / 2,
                                    (1 + spts[:, 1]) / 2])

            # Per-triangle vertex data
            tri_v = uverts[vids].swapaxes(0, 1)
            tri_n = vnorms[vids].swapaxes(0, 1)

            pts = _spherigon_smooth(pts, bary, tri_v, tri_n)

        return pts
