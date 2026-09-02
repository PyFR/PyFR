import numpy as np

from pyfr.quality import MeshQuality
from pyfr.shapes import interp_pts
from pyfr.writers.vtk.base import extra_field
from pyfr.writers.vtk.volume import BaseVolumeVTKWriter


class VTKMeshWriter(BaseVolumeVTKWriter):
    type = 'mesh'

    def __init__(self, meshf, pname=None, *, cfg=None, **kwargs):
        self.cfg = cfg

        # Build the face connectivity needed by the quality size ratios
        self.needs_con = cfg is not None

        super().__init__(meshf, pname, **kwargs)

    def process(self, outfname):
        self._load_mesh()
        self._write(outfname)

    def _load_mesh(self):
        self.mesh = self.reader.mesh

        # A bare mesh has no fields and no associated time
        self._vtk_vars, self._extra_fields = {}, {}
        self.tcurr = None

        if self.cfg is None:
            self._qcellf = {et: {} for et in self.mesh.spts}
        else:
            self._load_quality()

        # Default the per-etype divisor to the shape point order
        if self.divisor is None:
            etdivs = {et: self._spts_order(et) for et in self.mesh.spts}
        else:
            etdivs = {}
        self._init_etypes_div(self.divisor or 1, etdivs)

        self.einfo = [(etype, spts.shape[1])
                      for etype, spts in self.mesh.spts.items()]

    def _load_quality(self):
        quality = MeshQuality(self.mesh, self.cfg)
        self._qcellf = quality.cell_fields()

        # Emit one cell array per requested quality field
        dtype = np.dtype(self.dtype)
        want = set(self.fields or [])
        for name in quality.names:
            if not want or name in want:
                self._extra_fields[name] = extra_field(name, 'cell', 1,
                                                       dtype)

    def _prepare_pts(self, etype):
        spts = self.mesh.spts[etype].astype(self.dtype)

        # Calculate node locations of VTU elements
        vpts = interp_pts(self._mesh_op(etype), spts)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        return (vpts, None, self.mesh.spts_curved[etype],
                self._qcellf[etype], {})

    def _point_arrays(self, etype):
        return []
