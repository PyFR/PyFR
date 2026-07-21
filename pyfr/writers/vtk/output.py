import numpy as np

from pyfr.fields import CleanToGrid
from pyfr.mpiutil import mpi, scal_coll
from pyfr.util import first


class DirectVTKOutput:
    pyr_divisor_bump = 2

    def __init__(self, writer):
        self.writer = writer

    def npts(self, etype, neles, nsvpts):
        return neles*nsvpts

    def points(self, etype, vpts):
        return vpts.swapaxes(0, 1)

    def connectivity(self, etype, nodes, neles, nsvpts):
        con = np.tile(nodes, (neles, 1))
        con += (np.arange(neles)*nsvpts)[:, None]
        return con

    def point_fields(self, etype):
        for arr, dtype in self.writer._point_field_data(etype):
            yield arr.swapaxes(0, 1), dtype


class CleanToGridVTKOutput(DirectVTKOutput):
    pyr_divisor_bump = 0

    def __init__(self, writer):
        super().__init__(writer)

        cnodemap, svptsmap = writer._output_topology()
        shared = np.fromiter(writer.mesh.shared_nodes.by_node, dtype=int)
        self.cleaner = CleanToGrid(cnodemap, writer.etypes_div, svptsmap,
                                   shared)
        self.point_data = self._compute_fields()

    def _compute_fields(self):
        comm = self.cleaner.comm

        # Obtain the field data for each element type
        fdata = {}
        for etype in self.writer._prepared:
            fdata[etype] = self.writer._point_field_data(etype)

        # Agree on field count across ranks
        nfields = len(first(fdata.values(), ()))
        nfields = scal_coll(comm.Allreduce, nfields, op=mpi.MAX)

        # Iterate through the fields and average them
        out = []
        for i in range(nfields):
            efields, ncomp = {}, 0

            for etype, fields in fdata.items():
                arr, dt = fields[i]
                efields[etype] = arr.astype(dt, copy=False)
                ncomp = max(ncomp, arr.shape[2])

            ncomp = scal_coll(comm.Allreduce, ncomp, op=mpi.MAX)
            avg = self.cleaner.average(efields, ncomp, self.writer.dtype)
            out.append({et: (a, self.writer.dtype) for et, a in avg.items()})

        # Group by element type and return
        return {etype: [f[etype] for f in out]
                for etype in self.writer._prepared}

    def npts(self, etype, neles, nsvpts):
        return len(self.cleaner.layouts[etype][1])

    def points(self, etype, vpts):
        return self.cleaner.select(etype, vpts)

    def connectivity(self, etype, nodes, neles, nsvpts):
        return self.cleaner.layouts[etype][0][:, nodes]

    def point_fields(self, etype):
        yield from self.point_data[etype]
