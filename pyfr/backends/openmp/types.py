# -*- coding: utf-8 -*-

import collections
import itertools as it

import numpy as np

import pyfr.backends.base as base
from pyfr.util import lazyprop


class OpenMPMatrixBase(base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata.ctypes.data

        self.data = basedata[offset:offset + self.nrow*self.pitch]
        self.data = self.data.view(self.dtype).reshape(self.datashape)

        self.offset = offset

        # Pointer to our ndarray (used by ctypes)
        self._as_parameter_ = self.data.ctypes.data

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Trim any padding in the final dimension
        return self.data[...,:self.ioshape[-1]]

    def _set(self, ary):
        # Assign
        self.data[...,:ary.shape[-1]] = ary


class OpenMPMatrix(OpenMPMatrixBase, base.Matrix):
    def __init__(self, backend, ioshape, initval, extent, tags):
        super(OpenMPMatrix, self).__init__(backend, backend.fpdtype, ioshape,
                                           initval, extent, tags)


class OpenMPMatrixRSlice(base.MatrixRSlice):
    @lazyprop
    def data(self):
        # Since slices do not retain any information about the
        # high-order structure of an array it is fine to compact mat
        # down to two dimensions and simply slice this
        mat = self.parent
        return mat.data.reshape(mat.nrow, mat.leaddim)[self.p:self.q]

    @lazyprop
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
        if item.ktype == 'compute':
            item.run(*rtargs)
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

    def _exec_nonblock(self):
        while self._items:
            kern = self._items[0][0]

            # See if kern will block
            if self._at_sequence_point(kern) or kern.ktype == 'compute':
                break

            self._exec_item(*self._items.popleft())

    def _wait(self):
        if self._last and self._last.ktype == 'mpi':
            from mpi4py import MPI

            MPI.Prequest.Waitall(self._mpireqs)
            self._mpireqs = []

        self._last = None

    def _at_sequence_point(self, item):
        last = self._last

        return last and last.ktype == 'mpi' and item.ktype != 'mpi'

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
