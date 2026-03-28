from functools import cached_property

import numpy as np

import pyfr.backends.base as base


class _OpenCLMatrixCommon:
    @cached_property
    def _as_parameter_(self):
        return int(self.data)


class OpenCLMatrixBase(_OpenCLMatrixCommon, base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.offset = offset

        # If necessary, slice the buffer
        if offset:
            self.data = basedata.slice(offset, self.nbytes)
        else:
            self.data = basedata

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Get a pinned bounce buffer from the backend
        buf = self.backend.xfer_buf((self.nrow, self.leaddim), self.dtype)

        # Copy from device
        self.backend.queue.barrier()
        self.backend.cl.memcpy(self.backend.queue, buf, self.data, self.nbytes,
                               blocking=True)

        # Unpack and ensure we return owned data (not a view of the buffer)
        return np.require(self._unpack(buf), requirements='O')

    def _set(self, ary):
        # Pack into a pinned bounce buffer and copy to device
        buf = self.backend.xfer_buf((self.nrow, self.leaddim), self.dtype)
        self._pack(ary, out=buf)
        self.backend.queue.barrier()
        self.backend.cl.memcpy(self.backend.queue, self.data, buf, self.nbytes,
                               blocking=True)


class OpenCLMatrixSlice(_OpenCLMatrixCommon, base.MatrixSlice):
    @cached_property
    def data(self):
        if self.offset:
            nbytes = ((self.nrow - 1)*self.leaddim + self.ncol)*self.itemsize
            return self.basedata.slice(self.offset, nbytes)
        else:
            return self.basedata


class OpenCLMatrix(OpenCLMatrixBase, base.Matrix): pass
class OpenCLConstMatrix(OpenCLMatrixBase, base.ConstMatrix): pass
class OpenCLView(base.View): pass
class OpenCLXchgView(base.XchgView): pass


class OpenCLXchgMatrix(OpenCLMatrix, base.XchgMatrix):
    def __init__(self, backend, dtype, ioshape, initval, extent, tags):
        super().__init__(backend, dtype, ioshape, initval, extent, tags)

        # Allocate an empty buffer on the host for MPI to send/recv from
        shape = (self.nrow, self.ncol)
        self.hdata = backend.cl.pagelocked_empty(shape, dtype)


class OpenCLGraph(base.Graph):
    def _commit(self):
        # Map from kernels to event table locations
        evtidxs = {}

        # Kernel list complete with dependency information
        self.klist = klist = []

        for i, k in enumerate(self.knodes):
            evtidxs[k] = i

            # Resolve the event indices of kernels we depend on
            wait_evts = [evtidxs[dep] for dep in self._alldeps(k)] or None

            klist.append((k, wait_evts, k in self.depk))

        # Dependent MPI request list
        self.mreqlist = mreqlist = []

        for req, deps in zip(self.mpi_reqs, self.mpi_req_deps):
            if deps:
                mreqlist.append((req, [evtidxs[dep] for dep in deps]))

    def run(self, queue):
        events = [None]*len(self.klist)
        wait_for_events = self.backend.cl.wait_for_events

        # Submit the kernels to the queue
        for i, (k, wait_for, ret_evt) in enumerate(self.klist):
            if wait_for is not None:
                wait_for = [events[j] for j in wait_for]

            events[i] = k.run(queue, wait_for, ret_evt)

        # Flush the queue to ensure the kernels have started
        queue.flush()

        # Start all dependency-free MPI requests
        self._startall(self.mpi_root_reqs)

        # Start any remaining requests once their dependencies are satisfied
        for req, wait_for in self.mreqlist:
            wait_for_events([events[j] for j in wait_for])
            req.Start()

        # Wait for all of the MPI requests to finish
        self._waitall(self.mpi_reqs)
