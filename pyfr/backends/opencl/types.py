# -*- coding: utf-8 -*-

from functools import cached_property

import numpy as np
import pyopencl as cl

import pyfr.backends.base as base


class _OpenCLMatrixCommon(object):
    @cached_property
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
    @cached_property
    def data(self):
        nbytes = ((self.nrow - 1)*self.leaddim + self.ncol)*self.itemsize
        return self.basedata.get_sub_region(self.offset, nbytes)


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

        # OpenCL command queue
        self.cmd_q = cl.CommandQueue(backend.ctx)

        # Active copy event list
        self.copy_events = []

    def run(self, mpireqs=[]):
        # Start any MPI requests
        if mpireqs:
            self._startall(mpireqs)

        # Submit the kernels to the command queue
        for item, args, kwargs in self._items:
            item.run(self, *args, **kwargs)

        # If we started any MPI requests, wait for them
        if mpireqs:
            self.cmd_q.flush()
            self._waitall(mpireqs)

        # Wait for the kernels to finish and clear the queue
        self.cmd_q.finish()
        self.copy_events.clear()
        self._items.clear()
