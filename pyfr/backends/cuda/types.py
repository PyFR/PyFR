# -*- coding: utf-8 -*-

from ctypes import c_int, c_ssize_t, c_void_p, pythonapi, py_object

import numpy as np
import pycuda.driver as cuda

import pyfr.backends.base as base
from pyfr.util import lazyprop


_make_pybuf = pythonapi.PyMemoryView_FromMemory
_make_pybuf.argtypes = [c_void_p, c_ssize_t, c_int]
_make_pybuf.restype = py_object


class _CUDAMatrixCommon(object):
    @property
    def _as_parameter_(self):
        return self.data

    def __index__(self):
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
        cuda.memcpy_dtoh(buf, self.data)

        # Unpack
        return self._unpack(buf[:, :self.ncol])

    def _set(self, ary):
        # Allocate a new buffer with suitable padding and pack it
        buf = np.zeros((self.nrow, self.leaddim), dtype=self.dtype)
        buf[:, :self.ncol] = self._pack(ary)

        # Copy
        cuda.memcpy_htod(self.data, buf)


class CUDAMatrix(CUDAMatrixBase, base.Matrix):
    pass


class CUDAMatrixSlice(_CUDAMatrixCommon, base.MatrixSlice):
    def _init_data(self, mat):
        return (int(mat.basedata) + mat.offset +
                self.ra*self.pitch + self.ca*self.itemsize)


class CUDAMatrixBank(base.MatrixBank):
    def __index__(self):
        return self._curr_mat.data


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
            self.hdata = _make_pybuf(self.data, self.nbytes, 0x200)
        # Otherwise, allocate a buffer on the host for MPI to send/recv from
        else:
            self.hdata = cuda.pagelocked_empty((self.nrow, self.ncol),
                                               self.dtype, 'C')


class CUDAXchgView(base.XchgView):
    pass


class CUDAQueue(base.Queue):
    def __init__(self, backend):
        super().__init__(backend)

        # CUDA streams
        self.cuda_stream_comp = cuda.Stream()
        self.cuda_stream_copy = cuda.Stream()

    def _wait(self):
        last = self._last

        if last and last.ktype == 'compute':
            self.cuda_stream_comp.synchronize()
            self.cuda_stream_copy.synchronize()
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
            for q in filter(None, queues):
                q._exec_next()
                q._exec_nowait()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
