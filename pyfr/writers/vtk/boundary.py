from collections import defaultdict

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter


class VTKBoundaryWriter(BaseVTKWriter):
    type = 'boundary'
    dimensions = '2|3'
    output_curved = True
    needs_con = True   # snap.surface() reads mesh.bcon

    def __init__(self, meshf, boundaries, **kwargs):
        super().__init__(meshf, **kwargs)
        self.boundaries = boundaries
        if self.ndims != 3:
            raise RuntimeError('Boundary export only supported for 3D grids')

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        ecount = defaultdict(int)
        for bcname in self.boundaries:
            conn = self.mesh.bcon.get(bcname.removeprefix('bc/'))
            if conn is None:
                continue
            for etype, fidx, eidxs in conn.items():
                shapecls = subclass_where(BaseShape, name=etype)
                itype = shapecls.faces[fidx][0]
                ecount[itype] += len(eidxs)
        self.einfo = list(ecount.items())

    def _build_region(self):
        div = self.etypes_div[self.einfo[0][0]] if self.einfo else None
        return self._snap.surface(self.boundaries, divisor=div,
                                  refpts_fn=self._refpts_fn,
                                  clean=self._clean)

    def _cell_curved(self, itype):
        parts = [self.mesh.spts_curved[g[0]][g[2]]
                 for g in self._region._groups if g[3] == itype]
        return np.concatenate(parts) if parts else np.empty(0, dtype=bool)

    def _cell_aux(self, itype, fname):
        pieces = []
        for g in self._region._groups:
            if g[3] != itype:
                continue
            data = self.soln.aux.get(g[0], {}).get(fname)
            if data is None:
                continue
            pieces.append(data[g[2]])
        return np.concatenate(pieces) if pieces else None
