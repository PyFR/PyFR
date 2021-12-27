# -*- coding: utf-8 -*-

import pyfr.backends.base as base
from pyfr.util import lazyprop


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
    @lazyprop
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
    def run(self, mpireqs=[]):
        # Start any MPI requests
        if mpireqs:
            self._startall(mpireqs)

        # Run our kernels
        for item, args, kwargs in self._items:
            item.run(self, *args, **kwargs)

        # If we started any MPI requests, wait for them
        if mpireqs:
            self._waitall(mpireqs)

        # Clear the queue
        self._items.clear()
