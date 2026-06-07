import numpy as np


class DirectVTKOutput:
    # Element-major per-tri/per-cell emission with tile-pattern connectivity.
    # Used by STL (welded vertices expanded per-triangle) and spanwise (the
    # specialised averager owns its own output structure).
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
