import numpy as np

from pyfr.shapes import interp_pts
from pyfr.writers.vtk.volume import BaseVolumeVTKWriter


class VTKMeshWriter(BaseVolumeVTKWriter):
    type = 'mesh'

    def process(self, outfname):
        self._load_mesh()
        self._write(outfname)

    def _load_mesh(self):
        self.mesh = self.reader.mesh

        # A bare mesh has no fields and no associated time
        self._vtk_vars, self._extra_fields = {}, {}
        self.tcurr = None

        # Default the per-etype divisor to the shape point order
        if self.divisor is None:
            etdivs = {et: self._spts_order(et) for et in self.mesh.spts}
        else:
            etdivs = {}
        self._init_etypes_div(self.divisor or 1, etdivs)

        self.einfo = [(etype, spts.shape[1])
                      for etype, spts in self.mesh.spts.items()]

    def _prepare_pts(self, etype):
        spts = self.mesh.spts[etype].astype(self.dtype)

        # Calculate node locations of VTU elements
        vpts = interp_pts(self._mesh_op(etype), spts)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        return vpts, None, self.mesh.spts_curved[etype], {}, {}

    def _point_arrays(self, etype):
        return []
