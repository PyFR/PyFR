import numpy as np


class RegionVTKOutput:
    # Single VTK output adapter consuming a region + sample pair.  Region
    # carries the clean-vs-raw choice (clean=True dedups + averages pris/grads
    # at shared sub-points before providers run); this adapter just reads
    # whatever the region/sample expose.  Used by VTKVolumeWriter and
    # VTKBoundaryWriter.
    pyr_divisor_bump = 0

    def __init__(self, writer):
        self.writer = writer
        self.point_data = {et: list(writer._point_field_data(et))
                           for et, _ in writer.einfo}

    def npts(self, etype, neles, nsvpts):
        ploc = self.writer._region.ploc[etype]
        if self.writer._region.clean:
            return ploc.shape[1]
        return neles*nsvpts

    def points(self, etype, vpts):
        if self.writer._region.clean:
            return np.ascontiguousarray(vpts.T)
        return np.ascontiguousarray(
            vpts.transpose(2, 1, 0).reshape(-1, vpts.shape[0]))

    def connectivity(self, etype, nodes, neles, nsvpts):
        region = self.writer._region
        if region.clean:
            return region.cleaner.layouts[etype][0][:, nodes]
        con = np.tile(nodes, (neles, 1))
        con += (np.arange(neles)*nsvpts)[:, None]
        return con

    def point_fields(self, etype):
        yield from self.point_data[etype]


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
