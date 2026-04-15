import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.postproc.adapters import PostProcData
from pyfr.points import PointLocator, PointSampler
from pyfr.polys import TriPolyBasis
from pyfr.shapes import TriShape
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts


def _vertex_normals_mwa(fnorms, verts, vids):
    # Angle-weighted vertex normals (Thurmer & Wuthrich 1998);
    # weight each face normal by the interior angle at the vertex
    fnorms = fnorms / np.linalg.norm(fnorms, axis=1, keepdims=True)

    # v[ci] = vertex positions for corner ci of each face
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
    type = 'stl'
    output_curved = False

    def __init__(self, meshf, stlrgns, *, subdiv='linear', **kwargs):
        # Disable high-order output and (by default) subdivision
        kwargs['order'] = None
        kwargs.setdefault('divisor', 1)

        super().__init__(meshf, **kwargs)

        if self.ndims != 3:
            raise RuntimeError('STL export only supported for 3D grids')

        if subdiv not in ('linear', 'spherigon'):
            raise ValueError(f'Invalid subdiv type: {subdiv}')

        # Read and merge the STL surfaces from the mesh
        stl = np.vstack([self.reader.mesh.raw[f'regions/stl/{s}']
                         for s in stlrgns])

        # Subdivide the mesh
        pts = self._subdivide_pts(stl, self.divisor, subdiv)

        # Determine the unique vertices
        ppts, pinv = np.unique(pts.reshape(-1, 3), axis=0, return_inverse=True)

        # Locate these vertices in the mesh
        plocs = PointLocator(self.reader.mesh).locate(ppts)

        self._stl_pts = (pts, pinv, ppts, plocs)
        self._stl_aux_names = []

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        mesh, soln = self.mesh, self.soln
        pts, pinv, spts, slocs = self._stl_pts
        comm, rank, root = get_comm_rank_root()
        nsoln = len(self._soln_fields)

        # STL carries no per-element cell data; keep only point fields
        self._extra_fields = {n: m for n, m in self._extra_fields.items()
                              if m.kind == 'point'}

        # Aux fields sampled through the pipeline (those also in soln.aux)
        etype = self._extra_etype
        aux_info = [(n, m.ncomps) for n, m in self._extra_fields.items()
                    if n in soln.aux.get(etype, {})]

        naux = sum(n for _, n in aux_info)

        # Stack solution + DOF-sized aux into a single per-element array
        data = []
        for etype in mesh.eidxs:
            nupts, nvars, neles = soln.data[etype].shape
            arr = np.empty((nupts, nvars + naux, neles))
            arr[:, :nvars] = soln.data[etype]

            off = nvars
            for name, n in aux_info:
                a = np.atleast_3d(soln.aux[etype][name])
                arr[:, off:off + n] = a.transpose(1, 2, 0)
                off += n

            data.append(arr)

        # Create and configure a point sampler
        sampler = PointSampler(mesh, spts, slocs)
        sampler.configure_with_cfg_nvars(soln.config, nsoln + naux)

        # Perform the sampling
        samps = sampler.sample(data)

        # If we are the root rank then write out the triangle list
        if rank == root:
            samps = samps.swapaxes(0, 1)

            # Pre-process the solution fields only
            svars = self._pre_proc_fields(samps[:nsoln].astype(self.dtype))

            # Run postproc plugins at welded sample points
            pp_fields = {}
            if self.pp_plugins:
                adapter = PostProcData(self.cfg, self.soln, svars, spts.T)
                pp_fields = self._run_postprocs(adapter, self.pp_plugins)

            # Rebuild tri vertices from (possibly transformed) welded verts
            pts = spts[pinv].reshape(pts.shape)

            # Unpack and reshape solution onto STL triangles
            svars = svars[:, pinv].reshape(-1, *pts.shape[:2])
            svars = svars.swapaxes(0, 1)

            # Unpack aux fields onto STL triangles
            pointf = {}
            off = nsoln
            for name, n in aux_info:
                a = samps[off:off + n, pinv].astype(self.dtype)
                pointf[name] = a.reshape(n, *pts.shape[:2]).swapaxes(0, 1)
                off += n

            # Unpack postproc fields onto STL triangles
            for name, data in pp_fields.items():
                a = data[pinv]
                if a.ndim == 1:
                    pointf[name] = a.reshape(*pts.shape[:2])
                else:
                    pointf[name] = a.reshape(*pts.shape[:2], -1).swapaxes(1, 2)

            self.einfo = [('tri', pts.shape[1])]
            self._stl_info = pts, svars, pointf
        else:
            self.einfo = []

    def _prepare_pts(self, etype):
        pts, svars, pointf = self._stl_info
        return pts, svars, None, {}, pointf

    def _subdivide_pts(self, stl, order, subdiv):
        basis = TriPolyBasis(1, TriShape.std_ele(1))
        op = basis.nodal_basis_at(TriShape.std_ele(order))

        # Flat linear subdivision
        pts = interpolate_pts(op, stl[:, 1:].swapaxes(0, 1))

        if subdiv == 'spherigon' and order > 1:
            fnorms = stl[:, 0].astype(float)
            verts = stl[:, 1:].reshape(-1, 3)

            # Weld coincident vertices
            uverts, vids = np.unique(verts, axis=0, return_inverse=True)
            vids = vids.reshape(-1, 3)

            # Angle-weighted vertex normals
            vnorms = _vertex_normals_mwa(fnorms, uverts, vids)

            # Barycentric coordinates at the subdivision points
            spts = np.array(TriShape.std_ele(order))
            bary = np.column_stack([-(spts[:, 0] + spts[:, 1]) / 2,
                                    (1 + spts[:, 0]) / 2,
                                    (1 + spts[:, 1]) / 2])

            # Per-triangle vertex data
            tri_v = uverts[vids].swapaxes(0, 1)
            tri_n = vnorms[vids].swapaxes(0, 1)

            pts = _spherigon_smooth(pts, bary, tri_v, tri_n)

        return pts
