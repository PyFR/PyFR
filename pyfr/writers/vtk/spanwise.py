from pyfr.snapshot.spanwise import SpanwiseSnapshotRegion
from pyfr.writers.vtk.base import BaseVTKWriter


class VTKSpanwiseWriter(BaseVTKWriter):
    type = 'spanwise'
    output_curved = True
    needs_con = True
    dimensions = '3'

    def __init__(self, mesh, cfg, *, nstations=None, boundary=None,
                 periodic=None, **kwargs):
        if boundary is None and periodic is None:
            raise ValueError('Boundary or periodic must be specified')
        if boundary is not None and periodic is not None:
            raise ValueError('Boundary and periodic are mutually exclusive')

        if periodic is not None:
            self._source = ('periodic', periodic)
        else:
            parts = boundary.split(',')
            if len(parts) != 2:
                raise ValueError('Boundary must be a comma separated pair')
            self._source = ('boundary', parts[0].strip(), parts[1].strip())

        self._nstations = int(nstations) if nstations else None

        super().__init__(mesh, cfg, **kwargs)

    def _init_einfo(self):
        self.einfo = []

    def _build_region(self):
        return SpanwiseSnapshotRegion(self.mesh, self.cfg,
                                      source=self._source,
                                      nstations=self._nstations,
                                      refpts_fn=self._refpts_fn,
                                      clean=self._clean)

    def _post_sample(self, snap):
        if not self.einfo:
            self.einfo = [(it, self._region._ncells[it])
                          for it in self._region.etypes]

    def _emit_fields(self, kind):
        if kind == 'cell':
            return
        yield from super()._emit_fields(kind)
