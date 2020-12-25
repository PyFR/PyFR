# -*- coding: utf-8 -*-

import pyfr.backends.base as base
from pyfr.util import lazyprop

import numpy as np


class OpenMPMatrixBase(base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata.ctypes.data

        self.data = basedata[offset:offset + self.nrow*self.pitch]
        self.data = self.data.view(self.dtype)
        print('shape', self.data.shape)
        self.data = self.data.reshape(self.nblock, self.nrow, self.leaddim)
        print('shape', self.data.shape)

        self.offset = offset

        # Pointer to our ndarray (used by ctypes)
        self._as_parameter_ = self.data.ctypes.data

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        return self._unpack(self.data[:, :, :])

    def _set(self, ary):
        print('letsset', self.data.shape, ary.shape)
        self.data[:, :, :] = self._pack(ary)
        print('isitset', self.data.shape, ary.shape)

    def _pack(self, ary):
        #np.set_printoptions(threshold=np.inf)
        #print('pack', ary.ndim)
        #print('ary', self.tags, ary)
        # If necessary convert from SoA to AoAoSoA packing
        if ary.ndim > 2:
            n, k = ary.shape[-1], self.backend.soasz
            nvar, sz = ary.shape[-2], k*self.backend.aosoasz

            ary = np.pad(ary, [(0, 0)]*(ary.ndim - 1) + [(0, -n % sz)],
                         mode='constant')
            ary = ary.reshape(ary.shape[:-1] + (-1, k)).swapaxes(-2, -3)
            ary = ary.reshape(self.nrow, -1, sz*nvar).swapaxes(0, 1)
        else:
            n, sz = ary.shape[-1], self.backend.soasz*self.backend.aosoasz
            ary = np.pad(ary, [(0, 0)] + [(0, -n % self.nbcol)], mode='constant')
            ary = ary.reshape(self.nrow, -1, self.nbcol).swapaxes(0, 1)
            #ary = ary.reshape(-1, self.nrow, self.leaddim)

        #print('after packing')
        #print('ary', self.tags, ary)
        return np.ascontiguousarray(ary, dtype=self.dtype)

    def _unpack(self, ary):
        # If necessary unpack from AoSoA to SoA
        if len(self.ioshape) > 2:
            ary = ary.swapaxes(0, 1)
            ary = ary.reshape(self.datashape)
            ary = ary.swapaxes(-2, -3)
            ary = ary.reshape(self.ioshape[:-1] + (-1,))
            ary = ary[..., :self.ioshape[-1]]

        return ary


class OpenMPMatrix(OpenMPMatrixBase, base.Matrix):
    @lazyprop
    def hdata(self):
        return self.data


class OpenMPMatrixSlice(base.MatrixSlice):
    def _init_data(self, mat):
        return mat.data[self.ba:self.bb, self.ra:self.rb, :]

    @property
    def _as_parameter_(self):
        return self.data.ctypes.data


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
