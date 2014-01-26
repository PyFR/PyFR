# -*- coding: utf-8 -*-

import collections
import itertools as it

from mpi4py import MPI
import numpy as np

import pyfr.backends.base as base


class OpenMPMatrixBase(base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata.ctypes.data

        self.data = basedata[offset:offset + self.nrow*self.pitch]
        self.data = self.data.view(self.dtype).reshape(self.datashape)

        self.offset = offset

        # Process any initial value
        if self._initval is not None:
            self.set(self._initval)

        # Remove
        del self._initval

    def get(self):
        # Trim any padding in the final dimension
        return self.data[...,:self.ioshape[-1]]

    def set(self, ary):
        if ary.shape != self.ioshape:
            raise ValueError('Invalid matrix shape')

        # Assign
        self.data[...,:ary.shape[-1]] = ary

    @property
    def _as_parameter_(self):
        # Obtain a pointer to our ndarray
        return self.data.ctypes.data


class OpenMPMatrix(OpenMPMatrixBase, base.Matrix):
    def __init__(self, backend, ioshape, initval, extent, tags):
        super(OpenMPMatrix, self).__init__(backend, backend.fpdtype, ioshape,
                                           initval, extent, tags)


class OpenMPMatrixRSlice(base.MatrixRSlice):
    def __init__(self, backend, mat, p, q):
        super(OpenMPMatrixRSlice, self).__init__(backend, mat, p, q)

        # Since slices do not retain any information about the
        # high-order structure of an array it is fine to compact mat
        # down to two dimensions and simply slice this
        self.data = backend.compact_arr(mat.data)[p:q]

    @property
    def _as_parameter_(self):
        return self.data.ctypes.data


class OpenMPMatrixBank(base.MatrixBank):
    pass


class OpenMPConstMatrix(OpenMPMatrixBase, base.ConstMatrix):
    def __init__(self, backend, initval, extent, tags):
        super(OpenMPConstMatrix, self).__init__(backend, backend.fpdtype,
                                                initval.shape, initval,
                                                extent, tags)


class OpenMPMPIMatrix(OpenMPMatrix, base.MPIMatrix):
    pass


class OpenMPMPIView(base.MPIView):
    pass


class OpenMPView(base.View):
    def __init__(self, backend, matmap, rcmap, stridemap, vshape, tags):
        super(OpenMPView, self).__init__(backend, matmap, rcmap, stridemap,
                                         vshape, tags)

        self.mapping = OpenMPMatrixBase(backend, np.int32, (1, self.n),
                                        self.mapping, None, tags)

        if self.nvcol > 1:
            self.cstrides = OpenMPMatrixBase(backend, np.int32, (1, self.n),
                                             self.cstrides, None, tags)

        if self.nvrow > 1:
            self.rstrides = OpenMPMatrixBase(backend, np.int32, (1, self.n),
                                             self.rstrides, None, tags)


class OpenMPQueue(base.Queue):
    def __init__(self, backend):
        super(OpenMPQueue, self).__init__(backend)

        # Last kernel we executed
        self._last = None

        # Active MPI requests
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
        if base.iscomputekernel(item):
            item.run(*rtargs)
        elif base.ismpikernel(item):
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

    def _exec_nonblock(self):
        while self._items:
            kern = self._items[0][0]

            # See if kern will block
            if self._at_sequence_point(kern) or base.iscomputekernel(kern):
                break

            self._exec_item(*self._items.popleft())

    def _wait(self):
        if base.ismpikernel(self._last):
            MPI.Prequest.Waitall(self._mpireqs)
            self._mpireqs = []
        self._last = None

    def _at_sequence_point(self, item):
        if base.ismpikernel(self._last) and not base.ismpikernel(item):
            return True
        else:
            return False

    def run(self):
        while self._items:
            self._exec_next()
        self._wait()

    @staticmethod
    def runall(queues):
        # Fire off any non-blocking kernels
        for q in queues:
            q._exec_nonblock()

        while any(queues):
            # Execute a (potentially) blocking item from each queue
            for q in it.ifilter(None, queues):
                q._exec_nowait()

            # Now consider kernels which will wait
            for q in it.ifilter(None, queues):
                q._exec_next()
                q._exec_nonblock()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
