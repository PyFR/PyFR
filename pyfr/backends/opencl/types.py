# -*- coding: utf-8 -*-

import itertools as it

import numpy as np
import pyopencl as cl

import pyfr.backends.base as base
from pyfr.util import lazyprop


class OpenCLMatrixBase(base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.data = basedata.get_sub_region(offset, self.nrow*self.pitch + 1)
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
        cl.enqueue_copy(self.backend.qdflt, buf, self.data)

        # Slice to give the expected I/O shape
        return buf[...,:self.ioshape[-1]]

    def _set(self, ary):
        # Allocate a new buffer with suitable padding and assign
        buf = np.zeros(self.datashape, dtype=self.dtype)
        buf[...,:self.ioshape[-1]] = ary

        # Copy
        cl.enqueue_copy(self.backend.qdflt, self.data, buf)

    @property
    def _as_parameter_(self):
        return self.data.int_ptr


class OpenCLMatrix(OpenCLMatrixBase, base.Matrix):
    def __init__(self, backend, ioshape, initval, extent, tags):
        super(OpenCLMatrix, self).__init__(backend, backend.fpdtype, ioshape,
                                           initval, extent, tags)


class OpenCLMatrixRSlice(base.MatrixRSlice):
    @lazyprop
    def data(self):
        return self.parent.basedata.get_sub_region(self.offset,
                                                   self.nrow*self.pitch + 1)

    @property
    def _as_parameter_(self):
        return self.data.int_ptr


class OpenCLMatrixBank(base.MatrixBank):
    pass


class OpenCLConstMatrix(OpenCLMatrixBase, base.ConstMatrix):
    def __init__(self, backend, initval, extent, tags):
        super(OpenCLConstMatrix, self).__init__(backend, backend.fpdtype,
                                                initval.shape, initval,
                                                extent, tags)


class OpenCLView(base.View):
    def __init__(self, backend, matmap, rcmap, stridemap, vshape, tags):
        super(OpenCLView, self).__init__(backend, matmap, rcmap, stridemap,
                                       vshape, tags)

        self.mapping = OpenCLMatrixBase(backend, np.int32, (1, self.n),
                                        self.mapping, None, tags)

        if self.nvcol > 1:
            self.cstrides = OpenCLMatrixBase(backend, np.int32, (1, self.n),
                                             self.cstrides, None, tags)

        if self.nvrow > 1:
            self.rstrides = OpenCLMatrixBase(backend, np.int32, (1, self.n),
                                             self.rstrides, None, tags)


class OpenCLMPIMatrix(OpenCLMatrix, base.MPIMatrix):
    def __init__(self, backend, ioshape, initval, extent, tags):
        # Call the standard matrix constructor
        super(OpenCLMPIMatrix, self).__init__(backend, ioshape, initval,
                                              extent, tags)

        # Allocate an empty buffer on the host for MPI to send/recv from
        self.hdata = np.empty((self.nrow, self.ncol), self.dtype)


class OpenCLMPIView(base.MPIView):
    pass


class OpenCLQueue(base.Queue):
    def __init__(self, backend):
        super(OpenCLQueue, self).__init__(backend)

        # OpenCL command queues
        self.cl_queue_comp = cl.CommandQueue(backend.ctx)
        self.cl_queue_copy = cl.CommandQueue(backend.ctx)

    def _wait(self):
        last = self._last

        if last and last.ktype == 'compute':
            self.cl_queue_comp.finish()
            self.cl_queue_copy.finish()
        elif last and last.ktype == 'mpi':
            from mpi4py import MPI

            MPI.Prequest.Waitall(self.mpi_reqs)
            self.mpi_reqs = []

        self._last = None

    def _at_sequence_point(self, item):
        return self._last and self._last.ktype != item.ktype

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
