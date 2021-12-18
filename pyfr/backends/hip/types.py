# -*- coding: utf-8 -*-

import numpy as np

import pyfr.backends.base as base
from pyfr.util import make_pybuf


class _HIPMatrixCommon(object):
    @property
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
    def _init_data(self, mat):
        return (int(mat.basedata) + mat.offset +
                (self.ra*self.leaddim + self.ca)*self.itemsize)


class HIPMatrixBank(base.MatrixBank):
    pass


class HIPConstMatrix(HIPMatrixBase, base.ConstMatrix):
    pass


class HIPView(base.View):
    pass


class HIPXchgMatrix(HIPMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        # Call the standard matrix constructor
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # If MPI is HIP-aware then construct a buffer out of our HIP
        # device allocation and pass this directly to MPI
        if backend.mpitype == 'hip-aware':
            self.hdata = make_pybuf(self.data, self.nbytes, 0x200)
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
