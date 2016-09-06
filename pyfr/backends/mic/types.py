# -*- coding: utf-8 -*-

import numpy as np

import pyfr.backends.base as base


class MICMatrixBase(base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.data = basedata.dev_ptr + offset

        self.offset = offset

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Allocate an empty buffer
        buf = np.empty((self.nrow, self.leaddim), dtype=self.dtype)

        # Copy using the default stream
        self.backend.sdflt.transfer_device2host(
            self.basedata, buf.ctypes.data, self.nbytes,
            offset_device=self.offset
        )
        self.backend.sdflt.sync()

        # Unpack
        return self._unpack(buf[:, :self.ncol])

    def _set(self, ary):
        # Allocate a new buffer with suitable padding and pack it
        buf = np.zeros((self.nrow, self.leaddim), dtype=self.dtype)
        buf[:, :self.ncol] = self._pack(ary)

        # Copy using the default stream
        self.backend.sdflt.transfer_host2device(
            buf.ctypes.data, self.basedata, self.nbytes,
            offset_device=self.offset
        )
        self.backend.sdflt.sync()


class MICMatrix(MICMatrixBase, base.Matrix):
    pass


class MICMatrixRSlice(base.MatrixRSlice):
    @property
    def data(self):
        return self.basedata.dev_ptr + self.offset


class MICMatrixBank(base.MatrixBank):
    pass


class MICConstMatrix(MICMatrixBase, base.ConstMatrix):
    pass


class MICXchgMatrix(MICMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        # Call the standard matrix constructor
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # Allocate an empty buffer on the host for MPI to send/recv from
        self.hdata = np.empty((self.nrow, self.ncol), self.dtype)


class MICXchgView(base.XchgView):
    pass


class MICView(base.View):
    pass


class MICQueue(base.Queue):
    def __init__(self, backend):
        super().__init__(backend)

        # MIC stream
        self.mic_stream_comp = backend.sdflt

    def _exec_item(self, item, args, kwargs):
        item.run(self, *args, **kwargs)

        self.mic_stream_comp.sync()

        self._last = item

    def _exec_nonblock(self):
        while self._items:
            kern = self._items[0][0]

            # See if kern will block
            if self._at_sequence_point(kern) or kern.ktype == 'compute':
                break

            self._exec_item(*self._items.popleft())

    def _wait(self):
        if self._last and self._last.ktype == 'mpi':
            from mpi4py import MPI

            MPI.Prequest.Waitall(self.mpi_reqs)
            self.mpi_reqs = []

        self._last = None

    def _at_sequence_point(self, item):
        last = self._last

        return last and last.ktype == 'mpi' and item.ktype != 'mpi'

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
