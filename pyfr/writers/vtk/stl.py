import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.points import PointLocator, PointSampler
from pyfr.polys import TriPolyBasis
from pyfr.shapes import TriShape
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts


class VTKSTLWriter(BaseVTKWriter):
    type = 'stl'
    output_curved = False

    def __init__(self, meshf, stlrgns, **kwargs):
        # Disable high-order output and (by default) subdivision
        kwargs['order'] = None
        kwargs.setdefault('divisor', 1)

        super().__init__(meshf, **kwargs)

        if self.ndims != 3:
            raise RuntimeError('STL export only supported for 3D grids')

        # Read and merge the STL surfaces from the mesh
        pts = np.vstack([self.reader.mesh.raw[f'regions/stl/{s}'][:, 1:]
                         for s in stlrgns])

        # Subdivide the mesh
        pts = self._subdivide_pts(pts.swapaxes(0, 1), self.divisor)

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

        # Identify DOF-sized aux fields that can be interpolated
        for etype in mesh.eidxs:
            nupts = soln.data[etype].shape[0]
            anames = [name for name, arr in soln.aux.get(etype, {}).items()
                      if arr.shape[1:] == (nupts,)]
            break
        else:
            anames = []

        # Extend the solution arrays with DOF-sized aux for sampling
        data = []
        for etype in mesh.eidxs:
            nupts, nvars, neles = soln.data[etype].shape
            arr = np.empty((nupts, nvars + len(anames), neles))
            arr[:, :nvars] = soln.data[etype]
            for i, name in enumerate(anames, start=nvars):
                arr[:, i] = soln.aux[etype][name].T

            data.append(arr)

        # Create and configure a point sampler
        sampler = PointSampler(mesh, spts, slocs)
        sampler.configure_with_cfg_nvars(soln.config, nsoln + len(anames))

        # Perform the sampling
        samps = sampler.sample(data)

        # If we are the root rank then write out the triangle list
        if rank == root:
            samps = samps.swapaxes(0, 1)

            # Pre-process the solution fields only
            svars = self._pre_proc_fields(samps[:nsoln].astype(self.dtype))

            # Unpack and reshape solution onto STL triangles
            svars = svars[:, pinv].reshape(-1, *pts.shape[:2])
            svars = svars.swapaxes(0, 1)

            # Unpack aux fields
            pointf = {}
            for i, name in enumerate(anames):
                a = samps[nsoln + i, pinv]
                pointf[name] = a.reshape(1, *pts.shape[:2]).swapaxes(0, 1)

            self.einfo = [('tri', pts.shape[1])]
            self._stl_info = pts, svars, pointf
            self._stl_aux_names = anames
        else:
            self.einfo = []

    def _resolve_etype(self, etype):
        return self._extra_etype

    def _extra_field_lists(self, etype=None):
        return [], self._stl_aux_names

    def _prepare_pts(self, etype):
        pts, svars, pointf = self._stl_info
        return pts, svars, None, {}, pointf

    def _subdivide_pts(self, pts, order):
        basis = TriPolyBasis(2, TriShape.std_ele(1))
        op = basis.nodal_basis_at(TriShape.std_ele(order))

        return interpolate_pts(op, pts)
