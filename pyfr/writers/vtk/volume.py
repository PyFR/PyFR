from pyfr.snapshot import VolumeSnapshotRegion
from pyfr.writers.vtk.base import BaseVTKWriter


class VTKVolumeWriter(BaseVTKWriter):
    type = 'volume'
    dimensions = '2|3'
    output_curved = True

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        self.einfo = [(etype, self.soln.data[etype].shape[2])
                      for etype in self.mesh.eidxs]

    def _build_region(self):
        spec = [et for et, _ in self.einfo]
        div = self.etypes_div[spec[0]] if spec else None
        return VolumeSnapshotRegion(self._snap, spec, self._refpts_fn,
                                    divisor=div, clean=self._clean)
