# -*- coding: utf-8 -*-

from functools import cached_property

import numpy as np

import pyfr.backends.base as base


class _CUDAMatrixCommon:
    @cached_property
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
    @cached_property
    def data(self):
        return int(self.basedata) + self.offset


class CUDAConstMatrix(CUDAMatrixBase, base.ConstMatrix):
    pass


class CUDAView(base.View):
    pass


class CUDAXchgMatrix(CUDAMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        # Call the standard matrix constructor
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # If MPI is CUDA-aware then simply annotate our device buffer
        if backend.mpitype == 'cuda-aware':
            class HostData:
                __array_interface__ = {
                    'version': 3,
                    'typestr': np.dtype(self.dtype).str,
                    'data': (self.data, False),
                    'shape': (self.nrow, self.ncol)
                }

            self.hdata = np.array(HostData(), copy=False)
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

    def run(self, mpireqs=[]):
        # Start any MPI requests
        if mpireqs:
            self._startall(mpireqs)

        # Submit the kernels to the CUDA stream
        for item, args, kwargs in self._items:
            item.run(self, *args, **kwargs)

        # If we started any MPI requests, wait for them
        if mpireqs:
            self._waitall(mpireqs)

        # Wait for the kernels to finish and clear the queue
        self.stream.synchronize()
        self._items.clear()
