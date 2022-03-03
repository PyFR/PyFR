# -*- coding: utf-8 -*-

from functools import cached_property

import numpy as np

import pyfr.backends.base as base


class _HIPMatrixCommon:
    @cached_property
    def _as_parameter_(self):
        return self.data


class HIPMatrixBase(_HIPMatrixCommon, base.MatrixBase):
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
        self.backend.hip.memcpy(buf, self.data, self.nbytes)

        # Unpack
        return self._unpack(buf[None, :, :])

    def _set(self, ary):
        buf = self._pack(ary)

        # Copy
        self.backend.hip.memcpy(self.data, buf, self.nbytes)


class HIPMatrix(HIPMatrixBase, base.Matrix):
    pass


class HIPMatrixSlice(_HIPMatrixCommon, base.MatrixSlice):
    @cached_property
    def data(self):
        return int(self.basedata) + self.offset


class HIPConstMatrix(HIPMatrixBase, base.ConstMatrix):
    pass


class HIPView(base.View):
    pass


class HIPXchgMatrix(HIPMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        # Call the standard matrix constructor
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # If MPI is HIP-aware then simply annotate our device buffer
        if backend.mpitype == 'hip-aware':
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
            self.hdata = backend.hip.pagelocked_empty(shape, dtype)


class HIPXchgView(base.XchgView):
    pass


class HIPQueue(base.Queue):
    def __init__(self, backend):
        super().__init__(backend)

        # HIP stream
        self.stream = backend.hip.create_stream()

    def run(self, mpireqs=[]):
        # Start any MPI requests
        if mpireqs:
            self._startall(mpireqs)

        # Submit the kernels to the HIP stream
        for item, args, kwargs in self._items:
            item.run(self, *args, **kwargs)

        # If we started any MPI requests, wait for them
        if mpireqs:
            self._waitall(mpireqs)

        # Wait for the kernels to finish and clear the queue
        self.stream.synchronize()
        self._items.clear()
