import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts


class VTKVolumeWriter(BaseVTKWriter):
    type = 'volume'
    output_curved = True
    output_partition = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.einfo = [(etype, self.soln[etype].shape[2])
                      for etype in self.mesh.eidxs]

    def _prepare_pts(self, etype):
        spts = self.mesh.spts[etype].astype(self.dtype)
        soln = self.soln[etype].swapaxes(0, 1).astype(self.dtype)
        curved = self.mesh.spts_curved[etype]

        # Extract the partition number information
        part = self.soln[f'{etype}-parts']

        # Dimensions
        nspts, neles = spts.shape[:2]

        # Shape
        shapecls = subclass_where(BaseShape, name=etype)

        # Sub divison points inside of a standard element
        svpts = shapecls.std_ele(self.etypes_div[etype])
        nsvpts = len(svpts)

        if etype != 'pyr' and self.ho_output:
            svpts = [svpts[i] for i in self._nodemaps[etype, nsvpts]]

        # Generate the operator matrices
        soln_b = shapecls(nspts, self.cfg)
        mesh_vtu_op = soln_b.sbasis.nodal_basis_at(svpts)
        soln_vtu_op = soln_b.ubasis.nodal_basis_at(svpts)

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
