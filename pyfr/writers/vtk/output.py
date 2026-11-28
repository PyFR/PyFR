import numpy as np


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
        etypes = list(self.cleaner.layouts)

        # Obtain the field data for each element type
        fdata = {etype: self._point_arrays(etype) for etype in etypes}

        # Iterate through the fields and average them
        items = [({et: arrs[i] for et, arrs in fdata.items()}, dt)
                 for i, dt in enumerate(dtypes)]
        out = self.cleaner.average_items(items)

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
