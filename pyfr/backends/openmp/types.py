# -*- coding: utf-8 -*-

from functools import cached_property

import pyfr.backends.base as base


class OpenMPMatrixBase(base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata.ctypes.data

        self.data = basedata[offset:offset + self.nbytes]
        self.data = self.data.view(self.dtype)
        self.data = self.data.reshape(self.nblocks, self.nrow, self.leaddim)

        self.offset = offset

        # Pointer to our ndarray (used by ctypes)
        self._as_parameter_ = self.data.ctypes.data

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        return self._unpack(self.data)

    def _set(self, ary):
        self.data[:] = self._pack(ary)


class OpenMPMatrix(OpenMPMatrixBase, base.Matrix):
    @cached_property
    def hdata(self):
        return self.data


class OpenMPMatrixSlice(base.MatrixSlice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._as_parameter_map = {}

    def _init_data(self, mat):
        return mat.data[self.ba:self.bb, self.ra:self.rb, :]

    @property
    def _as_parameter_(self):
        try:
            return self._as_parameter_map[self.parent.mid]
        except KeyError:
            param = self.data.ctypes.data
            self._as_parameter_map[self.parent.mid] = param

            return param


class OpenMPMatrixBank(base.MatrixBank):
    pass


class OpenMPConstMatrix(OpenMPMatrixBase, base.ConstMatrix):
    pass


class OpenMPXchgMatrix(OpenMPMatrix, base.XchgMatrix):
    pass


class OpenMPXchgView(base.XchgView):
    pass


class OpenMPView(base.View):
    pass


class OpenMPQueue(base.Queue):
    def _exec_nonblock(self):
        while self._items:
            kern = self._items[0][0]

            # See if kern will block
            if self._at_sequence_point(kern) or kern.ktype == 'compute':
                break

            self._exec_item(*self._items.popleft())

    def _wait(self):
        if self._last_ktype == 'mpi':
            from mpi4py import MPI

            MPI.Prequest.Waitall(self.mpi_reqs)
            self.mpi_reqs = []

        self._last_ktype = None

    def _at_sequence_point(self, item):
        return self._last_ktype == 'mpi' and item.ktype != 'mpi'

    @staticmethod
    def runall(queues):
        # Fire off any non-blocking kernels
        for q in queues:
            q._exec_nonblock()

        while any(queues):
            # Execute a (potentially) blocking item from each queue
            for q in filter(None, queues):
                q._exec_nowait()

            # Now consider kernels which will wait
            for q in filter(None, queues):
                q._exec_next()
                q._exec_nonblock()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
