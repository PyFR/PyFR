# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

import pyfr.backends.base as base


class _OpenCLMatrixCommon(object):
    @property
    def _as_parameter_(self):
        return self.data.int_ptr


class OpenCLMatrixBase(_OpenCLMatrixCommon, base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.data = basedata.get_sub_region(offset, self.nbytes)
        self.offset = offset

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Allocate an empty buffer
        buf = np.empty((self.nrow, self.leaddim), dtype=self.dtype)

        # Copy
        cl.enqueue_copy(self.backend.qdflt, buf, self.data)

        # Unpack
        return self._unpack(buf[None, :, :])

    def _set(self, ary):
        buf = self._pack(ary)

        # Copy
        cl.enqueue_copy(self.backend.qdflt, self.data, buf)


class OpenCLMatrix(OpenCLMatrixBase, base.Matrix):
    pass


class OpenCLMatrixSlice(_OpenCLMatrixCommon, base.MatrixSlice):
    def _init_data(self, mat):
        start = (self.ra*self.leaddim + self.ca)*self.itemsize
        nbytes = ((self.nrow - 1)*self.leaddim + self.ncol)*self.itemsize

        return mat.basedata.get_sub_region(mat.offset + start, nbytes)


class OpenCLMatrixBank(base.MatrixBank):
    pass


class OpenCLConstMatrix(OpenCLMatrixBase, base.ConstMatrix):
    pass


class OpenCLView(base.View):
    pass


class OpenCLXchgMatrix(OpenCLMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # Allocate an empty buffer on the host for MPI to send/recv from
        self.hdata = np.empty((self.nrow, self.ncol), self.dtype)


class OpenCLXchgView(base.XchgView):
    pass


class OpenCLQueue(base.Queue):
    def __init__(self, backend):
        super().__init__(backend)

        # OpenCL command queues
        self.cmd_q_comp = cl.CommandQueue(backend.ctx)
        self.cmd_q_copy = cl.CommandQueue(backend.ctx)

        # Active copy event list
        self.copy_events = []

    def _wait(self):
        if self._last_ktype == 'compute':
            self.cmd_q_comp.finish()
            self.cmd_q_copy.finish()
            self.copy_events.clear()
        elif self._last_ktype == 'mpi':
            from mpi4py import MPI

            MPI.Prequest.Waitall(self.mpi_reqs)
            self.mpi_reqs = []

        self._last_ktype = None

    def _at_sequence_point(self, item):
        return self._last_ktype != item.ktype

    @staticmethod
    def runall(queues):
        # First run any items which will not result in an implicit wait
        for q in queues:
            q._exec_nowait()

        # So long as there are items remaining in the queues
        while any(queues):
            # Execute a (potentially) blocking item from each queue
            for q in filter(None, queues):
                q._exec_next()
                q._exec_nowait()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
