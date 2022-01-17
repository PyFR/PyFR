# -*- coding: utf-8 -*-

from ctypes import addressof, c_void_p, cast
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
    @cached_property
    def data(self):
        return self.parent.data[self.ba:self.bb, self.ra:self.rb, :]

    @cached_property
    def _as_parameter_(self):
        return self.data.ctypes.data


class OpenMPConstMatrix(OpenMPMatrixBase, base.ConstMatrix): pass
class OpenMPXchgMatrix(OpenMPMatrix, base.XchgMatrix): pass
class OpenMPXchgView(base.XchgView): pass
class OpenMPView(base.View): pass


class OpenMPGraph(base.Graph):
    def __init__(self, backend):
        super().__init__(backend)

        self.klist = []

    def commit(self):
        super().commit()

        self._nkerns = n = len(self.klist)

        # Obtain pointers to our kernel functions
        self._kfuns = [cast(k.fun, c_void_p) for k in self.klist]
        self._kfuns = (c_void_p * n)(*self._kfuns)

        # Obtain pointers to their corresponding arguments
        self._kargs = [addressof(k.kargs) for k in self.klist]
        self._kargs = (c_void_p * n)(*self._kargs)

    def run(self):
        self.backend.krunner(self._nkerns, self._kfuns, self._kargs)

