# -*- coding: utf-8 -*-

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
    pass


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
    pass


class OpenMPMPIMatrix(OpenMPMatrix, base.MPIMatrix):
    pass


class OpenMPMPIView(base.MPIView):
    pass


class OpenMPView(base.View):
    def __init__(self, backend, matmap, rcmap, stridemap, vshape, tags):
        super(OpenMPView, self).__init__(backend, matmap, rcmap, stridemap,
                                         vshape, tags)

        self.mapping = OpenMPMatrixBase(backend, np.int32, (1, self.n),
                                        self.mapping, None, None, tags)

        if self.nvcol > 1:
            self.cstrides = OpenMPMatrixBase(backend, np.int32, (1, self.n),
                                             self.cstrides, None, None, tags)

        if self.nvrow > 1:
            self.rstrides = OpenMPMatrixBase(backend, np.int32, (1, self.n),
                                             self.rstrides, None, None, tags)


class OpenMPQueue(base.Queue):
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

            MPI.Prequest.Waitall(self.mpi_reqs)
            self.mpi_reqs = []

        self._last = None

    def _at_sequence_point(self, item):
        last = self._last

        return last and last.ktype == 'mpi' and item.ktype != 'mpi'

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
