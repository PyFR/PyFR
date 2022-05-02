# -*- coding: utf-8 -*-

from collections import defaultdict

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
        self.mpi_idxs = defaultdict(list)

    def add_mpi_req(self, req, deps=[]):
        super().add_mpi_req(req, deps)

        if deps:
            ix = max(self.knodes[d] for d in deps)

            self.mpi_idxs[ix].append(req)

    def commit(self):
        super().commit()

        n = len(self.klist)

        # Obtain pointers to our kernel functions and their arguments
        self._kfunargs = (c_void_p * (2*n))()
        self._kfunargs[0::2] = [cast(k.fun, c_void_p) for k in self.klist]
        self._kfunargs[1::2] = [addressof(k.kargs) for k in self.klist]

        # Group kernels in runs separated by MPI requests
        self._runlist, i = [], 0

        for j in sorted(self.mpi_idxs):
            self._runlist.append((i, j - i, self.mpi_idxs[j]))
            i = j

        if i != n - 1:
            self._runlist.append((i, n - i, []))

    def run(self):
        from mpi4py import MPI

        # Start all dependency-free MPI requests
        MPI.Prequest.Startall(self.mpi_root_reqs)

        for i, n, reqs in self._runlist:
            self.backend.krunner(i, n, self._kfunargs)

            MPI.Prequest.Startall(reqs)

        # Wait for all of the MPI requests to finish
        MPI.Prequest.Waitall(self.mpi_reqs)
