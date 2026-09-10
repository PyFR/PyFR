import numpy as np

from pyfr.mpiutil import mpi, scal_coll


class DirectVTKOutput:
    pyr_divisor_bump = 2

    def __init__(self, point_arrays, dtypes, make_cleaner):
        self._point_arrays = point_arrays

    def npts(self, etype, neles, nsvpts):
        return neles*nsvpts

    def points(self, etype, vpts):
        return vpts.swapaxes(0, 1)

    def connectivity(self, etype, nodes, neles, nsvpts):
        con = np.tile(nodes, (neles, 1))
        con += (np.arange(neles)*nsvpts)[:, None]
        return con

    def point_arrays(self, etype):
        return [arr.swapaxes(0, 1) for arr in self._point_arrays(etype)]


class CleanToGridVTKOutput(DirectVTKOutput):
    pyr_divisor_bump = 0

    def __init__(self, point_arrays, dtypes, make_cleaner):
        super().__init__(point_arrays, dtypes, make_cleaner)

        self.cleaner = make_cleaner()
        self.point_data = self._average(dtypes)

    def _average(self, dtypes):
        comm, etypes = self.cleaner.comm, list(self.cleaner.layouts)

        # Obtain the field data for each element type
        fdata = {etype: self._point_arrays(etype) for etype in etypes}

        # Iterate through the fields and average them
        out = []
        for i, dtype in enumerate(dtypes):
            efields, ncomp = {}, 0

            for etype, arrays in fdata.items():
                efields[etype] = arrays[i].astype(dtype, copy=False)
                ncomp = max(ncomp, arrays[i].shape[2])

            ncomp = scal_coll(comm.Allreduce, ncomp, op=mpi.MAX)
            out.append(self.cleaner.average(efields, ncomp, dtype))

        # Group by element type and return
        return {etype: [f[etype] for f in out] for etype in etypes}

    def npts(self, etype, neles, nsvpts):
        return len(self.cleaner.layouts[etype][1])

    def points(self, etype, vpts):
        return self.cleaner.select(etype, vpts)

    def connectivity(self, etype, nodes, neles, nsvpts):
        return self.cleaner.layouts[etype][0][:, nodes]

    def point_arrays(self, etype):
        return self.point_data[etype]
