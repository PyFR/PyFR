from collections import defaultdict

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.shapes import BaseShape
from pyfr.snapshot.region import SurfaceSnapshotRegion
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter


class VTKBoundaryWriter(BaseVTKWriter):
    type = 'boundary'
    dimensions = '2|3'
    output_curved = True
    needs_con = True

    def __init__(self, mesh, cfg, *, boundaries, **kwargs):
        if not boundaries:
            raise ValueError('Boundary export requires at least one boundary')
        self.boundaries = list(boundaries)
        super().__init__(mesh, cfg, **kwargs)
        if mesh.ndims != 3:
            raise RuntimeError('Boundary export only supported for 3D grids')

    def _init_einfo(self):
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

        comm, _, _ = get_comm_rank_root()
        local = sum(n for _, n in self.einfo)
        if comm.allreduce(local, op=mpi.SUM) == 0:
            raise ValueError(
                f'No boundary elements found for {self.boundaries!r}')

    def _build_region(self):
        div = self.etypes_div[self.einfo[0][0]] if self.einfo else None
        return SurfaceSnapshotRegion(self.mesh, self.cfg, self.boundaries,
                                     div, refpts_fn=self._refpts_fn,
                                     clean=self._clean)
