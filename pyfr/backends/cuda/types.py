# -*- coding: utf-8 -*-

import numpy as np

import pyfr.backends.base as base
from pyfr.util import make_pybuf


class _CUDAMatrixCommon(object):
    @property
    def _as_parameter_(self):
        return self.data


class CUDAMatrixBase(_CUDAMatrixCommon, base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.data = int(self.basedata) + offset
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
        self.backend.cuda.memcpy(buf, self.data, self.nbytes)

        # Unpack
        return self._unpack(buf[None, :, :])

    def _set(self, ary):
        buf = self._pack(ary)

        # Copy
        self.backend.cuda.memcpy(self.data, buf, self.nbytes)


class CUDAMatrix(CUDAMatrixBase, base.Matrix):
    pass


class CUDAMatrixSlice(_CUDAMatrixCommon, base.MatrixSlice):
    def _init_data(self, mat):
        return (int(mat.basedata) + mat.offset +
                (self.ra*self.leaddim + self.ca)*self.itemsize)


class CUDAMatrixBank(base.MatrixBank):
    pass


class CUDAConstMatrix(CUDAMatrixBase, base.ConstMatrix):
    pass


class CUDAView(base.View):
    pass


class CUDAXchgMatrix(CUDAMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        # Call the standard matrix constructor
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # If MPI is CUDA-aware then construct a buffer out of our CUDA
        # device allocation and pass this directly to MPI
        if backend.mpitype == 'cuda-aware':
            self.hdata = make_pybuf(self.data, self.nbytes, 0x200)
        # Otherwise, allocate a buffer on the host for MPI to send/recv from
        else:
            shape, dtype = (self.nrow, self.ncol), self.dtype
            self.hdata = backend.cuda.pagelocked_empty(shape, dtype)


class CUDAXchgView(base.XchgView):
    pass


class CUDAQueue(base.Queue):
    def __init__(self, backend):
        super().__init__(backend)

        # CUDA stream
        self.stream = backend.cuda.create_stream()

    def _wait(self):
        if self._last_ktype == 'compute':
            self.stream.synchronize()
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
