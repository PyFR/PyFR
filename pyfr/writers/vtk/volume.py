import numpy as np

from pyfr.cache import memoize
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts


class VTKVolumeWriter(BaseVTKWriter):
    type = 'volume'
    output_curved = True
    output_partition = True

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        self.einfo = [(etype, self.soln[etype].shape[2])
                      for etype in self.mesh.eidxs]

    @memoize
    def _opmats(self, etype, cfg):
        # Shape
        shapecls = subclass_where(BaseShape, name=etype)

        # Sub divison points inside of a standard element
        svpts = shapecls.std_ele(self.etypes_div[etype])
        nsvpts = len(svpts)

        # Basis
        basis = shapecls(len(self.mesh.spts[etype]), cfg)

        if etype != 'pyr' and self.ho_output:
            svpts = [svpts[i] for i in self._nodemaps[etype, nsvpts]]

        mesh_op = basis.sbasis.nodal_basis_at(svpts)
        soln_op = basis.ubasis.nodal_basis_at(svpts)

        return mesh_op, soln_op

    def _prepare_pts(self, etype):
        spts = self.mesh.spts[etype].astype(self.dtype)
        soln = self.soln[etype].swapaxes(0, 1).astype(self.dtype)
        curved = self.mesh.spts_curved[etype]

        # Extract the partition number information
        part = self.soln[f'{etype}-parts']

        # Generate the interpolation operator matrices
        mesh_vtu_op, soln_vtu_op = self._opmats(etype, self.cfg)

        # Calculate node locations of VTU elements
        vpts = interpolate_pts(mesh_vtu_op, spts)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        # Pre-process the solution
        soln = self._pre_proc_fields(soln).swapaxes(0, 1)

        # Interpolate the solution to the vis points
        vsoln = interpolate_pts(soln_vtu_op, soln)

        return vpts, vsoln, curved, part
