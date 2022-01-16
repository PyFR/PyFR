# -*- coding: utf-8 -*-

from functools import cached_property

import numpy as np

import pyfr.backends.base as base


class _OpenCLMatrixCommon:
    @cached_property
    def _as_parameter_(self):
        return int(self.data)


class OpenCLMatrixBase(_OpenCLMatrixCommon, base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.offset = offset

        # If necessary, slice the buffer
        if offset:
            self.data = basedata.slice(offset, self.nbytes)
        else:
            self.data = basedata

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Allocate an empty buffer
        buf = np.empty((self.nrow, self.leaddim), dtype=self.dtype)

        # Copy
        self.backend.cl.memcpy(buf, self.data, self.nbytes)

        # Unpack
        return self._unpack(buf[None, :, :])

    def _set(self, ary):
        buf = self._pack(ary)

        # Copy
        self.backend.cl.memcpy(self.data, buf, self.nbytes)


class OpenCLMatrix(OpenCLMatrixBase, base.Matrix):
    pass


class OpenCLMatrixSlice(_OpenCLMatrixCommon, base.MatrixSlice):
    @cached_property
    def data(self):
        if self.offset:
            nbytes = ((self.nrow - 1)*self.leaddim + self.ncol)*self.itemsize
            return self.basedata.slice(self.offset, nbytes)
        else:
            return self.basedata


class OpenCLConstMatrix(OpenCLMatrixBase, base.ConstMatrix):
    pass


class OpenCLView(base.View):
    pass


class OpenCLXchgMatrix(OpenCLMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # Allocate an empty buffer on the host for MPI to send/recv from
        shape, dtype = (self.nrow, self.ncol), self.dtype
        self.hdata = backend.cl.pagelocked_empty(shape, dtype)


class OpenCLXchgView(base.XchgView):
    pass


class OpenCLOrderedMetaKernel(base.MetaKernel):
    pass


class OpenCLUnorderedMetaKernel(base.MetaKernel):
    pass
