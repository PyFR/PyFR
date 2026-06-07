from pyfr.snapshot import VolumeSnapshotRegion
from pyfr.writers.vtk.base import BaseVTKWriter


class VTKVolumeWriter(BaseVTKWriter):
    type = 'volume'
    dimensions = '2|3'
    output_curved = True

    def _init_einfo(self):
        self.einfo = [(et, len(self.mesh.eidxs[et])) for et in self.mesh.eidxs]

    def _build_region(self):
        spec = {et: slice(None) for et, _ in self.einfo}
        div = self.etypes_div[next(iter(spec))] if spec else None
        return VolumeSnapshotRegion(self.mesh, self.cfg, spec, self._refpts_fn,
                                    divisor=div, clean=self._clean)
