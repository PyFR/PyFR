# -*- coding: utf-8 -*-

import collections
import itertools as it

import numpy as np
import pycuda.driver as cuda

import pyfr.backends.base as base


class CUDAMatrixBase(base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = int(basedata)
        self.data = self.basedata + offset
        self.offset = offset

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Allocate an empty buffer
        buf = np.empty(self.datashape, dtype=self.dtype)

        # Copy
        cuda.memcpy_dtoh(buf, self.data)

        # Slice to give the expected I/O shape
        return buf[...,:self.ioshape[-1]]

    def _set(self, ary):
        # Allocate a new buffer with suitable padding and assign
        buf = np.zeros(self.datashape, dtype=self.dtype)
        buf[...,:self.ioshape[-1]] = ary

        # Copy
        cuda.memcpy_htod(self.data, buf)

    @property
    def _as_parameter_(self):
        return self.data

    def __long__(self):
        return self.data


class CUDAMatrix(CUDAMatrixBase, base.Matrix):
    def __init__(self, backend, ioshape, initval, extent, tags):
        super(CUDAMatrix, self).__init__(backend, backend.fpdtype, ioshape,
                                         initval, extent, tags)


class CUDAMatrixRSlice(base.MatrixRSlice):
    @property
    def _as_parameter_(self):
        return self.parent.basedata + self.offset

    def __long__(self):
        return self.parent.basedata + self.offset


class CUDAMatrixBank(base.MatrixBank):
    def __long__(self):
        return self._curr_mat.data


class CUDAConstMatrix(CUDAMatrixBase, base.ConstMatrix):
    def __init__(self, backend, initval, extent, tags):
        ioshape = initval.shape
        super(CUDAConstMatrix, self).__init__(backend, backend.fpdtype,
                                              ioshape, initval, extent, tags)

class CUDAView(base.View):
    def __init__(self, backend, matmap, rcmap, stridemap, vshape, tags):
        super(CUDAView, self).__init__(backend, matmap, rcmap, stridemap,
                                       vshape, tags)

        self.mapping = CUDAMatrixBase(backend, np.int32, (1, self.n),
                                      self.mapping, None, tags)

        if self.nvcol > 1:
            self.cstrides = CUDAMatrixBase(backend, np.int32, (1, self.n),
                                           self.cstrides, None, tags)

        if self.nvrow > 1:
            self.rstrides = CUDAMatrixBase(backend, np.int32, (1, self.n),
                                           self.rstrides, None, tags)


class CUDAMPIMatrix(CUDAMatrix, base.MPIMatrix):
    def __init__(self, backend, ioshape, initval, extent, tags):
        # Call the standard matrix constructor
        super(CUDAMPIMatrix, self).__init__(backend, ioshape, initval, extent,
                                            tags)

        # Allocate a page-locked buffer on the host for MPI to send/recv from
        self.hdata = cuda.pagelocked_empty((self.nrow, self.ncol),
                                           self.dtype, 'C')


class CUDAMPIView(base.MPIView):
    pass


class CUDAQueue(base.Queue):
    def __init__(self, backend):
        super(CUDAQueue, self).__init__(backend)

        # Last kernel we executed
        self._last = None

        # CUDA stream and MPI request list
        self._stream_comp = cuda.Stream()
        self._stream_copy = cuda.Stream()
        self._mpireqs = []

        # Items waiting to be executed
        self._items = collections.deque()

    def __lshift__(self, items):
        self._items.extend(items)

    def __mod__(self, items):
        self.run()
        self << items
        self.run()

    def __nonzero__(self):
        return bool(self._items)

    def _exec_item(self, item, rtargs):
        if item.ktype == 'compute':
            item.run(self._stream_comp, self._stream_copy, *rtargs)
        elif item.ktype == 'mpi':
            item.run(self._mpireqs, *rtargs)
        else:
            raise ValueError('Non compute/MPI kernel in queue')
        self._last = item

    def _exec_next(self):
        item, rtargs = self._items.popleft()

        # If we are at a sequence point then wait for current items
        if self._at_sequence_point(item):
            self._wait()

        # Execute the item
        self._exec_item(item, rtargs)

    def _exec_nowait(self):
        while self._items and not self._at_sequence_point(self._items[0][0]):
            self._exec_item(*self._items.popleft())

    def _wait(self):
        last = self._last

        if last and last.ktype == 'compute':
            self._stream_comp.synchronize()
            self._stream_copy.synchronize()
        elif last and last.ktype == 'mpi':
            from mpi4py import MPI

            MPI.Prequest.Waitall(self._mpireqs)
            self._mpireqs = []

        self._last = None

    def _at_sequence_point(self, item):
        return self._last and self._last.ktype != item.ktype

    def run(self):
        while self._items:
            self._exec_next()
        self._wait()

    @staticmethod
    def runall(queues):
        # First run any items which will not result in an implicit wait
        for q in queues:
            q._exec_nowait()

        # So long as there are items remaining in the queues
        while any(queues):
            # Execute a (potentially) blocking item from each queue
            for q in it.ifilter(None, queues):
                q._exec_next()
                q._exec_nowait()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
