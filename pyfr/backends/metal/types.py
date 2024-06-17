from collections import defaultdict
from functools import cached_property

import numpy as np

import pyfr.backends.base as base


class MetalMatrixBase(base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.offset = offset
        self.data = (self.basedata, self.offset)

        # Map the buffer
        buf = basedata.contents().as_buffer(offset + self.nbytes)
        self.hdata = np.frombuffer(buf, dtype=self.dtype, offset=self.offset)

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Ensure the host buffer is in sync with the device
        cbuf = self.backend.queue.commandBuffer()
        blit = cbuf.blitCommandEncoder()
        blit.synchronizeResource_(self.basedata)
        blit.endEncoding()
        cbuf.commit()
        cbuf.waitUntilCompleted()

        self.backend.last_cbuf = None

        # Unpack
        return self._unpack(self.hdata).copy()

    def _set(self, ary):
        # Wait for any outstanding work to finish
        self.backend.wait()

        # Update the host buffer contents
        self.hdata[:] = self._pack(ary).flat

        # Inform Metal about the update
        self.basedata.didModifyRange_((self.offset, self.nbytes))


class MetalMatrixSlice(base.MatrixSlice):
    @cached_property
    def data(self):
        return self.basedata, self.offset


class MetalMatrix(MetalMatrixBase, base.Matrix): pass
class MetalConstMatrix(MetalMatrixBase, base.ConstMatrix): pass
class MetalView(base.View): pass
class MetalXchgView(base.XchgView): pass
class MetalXchgMatrix(MetalMatrix, base.XchgMatrix): pass


class MetalGraph(base.Graph):
    needs_pdeps = False

    def __init__(self, backend):
        super().__init__(backend)

        self.klist = []
        self.mpi_idxs = defaultdict(list)

    def add_mpi_req(self, req, deps=[]):
        super().add_mpi_req(req, deps)

        if deps:
            ix = max(self.knodes[d] for d in deps)

            self.mpi_idxs[ix].append(req)

    def commit(self):
        super().commit()

        n = len(self.klist)

        # Group kernels in runs separated by MPI requests
        self._kerns, self._mreqs, i = [], [], 0

        for j in sorted(self.mpi_idxs):
            self._kerns.append(self.klist[i:j])
            self._mreqs.append(self.mpi_idxs[j])
            i = j

        if i != n - 1:
            self._kerns.append(self.klist[i:])

    def run(self, queue):
        cbufs = []

        # Submit the kernels to the queue
        for kerns in self._kerns:
            cbuf = queue.commandBuffer()
            for k in kerns:
                k.run(cbuf)

            cbuf.commit()
            cbufs.append(cbuf)

        # Start all dependency-free MPI requests
        self._startall(self.mpi_root_reqs)

        # Start any remaining requests once their dependencies are satisfied
        for cbuf, reqs in zip(cbufs, self._mreqs):
            cbuf.waitUntilCompleted()
            self._startall(reqs)

        # Wait for all of the MPI requests to finish
        self._waitall(self.mpi_reqs)

        # Return the last-submitted command buffer
        return cbufs[-1]
