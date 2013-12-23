# -*- coding: utf-8 -*-

import collections
import itertools as it

from mpi4py import MPI
import numpy as np

import pyfr.backends.base as base
from pyfr.nputil import npaligned


class OpenMPMatrixBase(base.MatrixBase):
    def __init__(self, backend, dtype, ioshape, initval, tags):
        super(OpenMPMatrixBase, self).__init__(backend, ioshape, tags)

        # Data type info
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize

        # Types must be multiples of 32
        assert (32 % self.itemsize) == 0

        # Alignment requirement for the final dimension
        ldmod = 32 // self.itemsize if 'align' in tags else 1

        # Our shape and dimensionality
        shape, ndim = list(self.ioshape), len(ioshape)

        if ndim == 2:
            nrow, ncol = shape
        elif ndim == 3 or ndim == 4:
            nrow = shape[0] if ndim == 3 else shape[0]*shape[1]
            ncol = shape[-2]*shape[-1] + (1 - shape[-2])*(shape[-1] % -ldmod)

        # Pad the final dimension
        shape[-1] -= shape[-1] % -ldmod

        # Assign
        self.nrow, self.ncol = nrow, ncol
        self.leaddim = ncol - (ncol % -ldmod)
        self.leadsubdim = shape[-1]

        # Allocate, ensuring data is on a 32-byte boundary (this is
        # separate to the dimension alignment above)
        self.data = npaligned(shape, dtype=self.dtype)

        # Process any initial value
        if initval is not None:
            self.set(initval)

    def get(self):
        # Trim any padding in the final dimension
        return self.data[...,:self.ioshape[-1]]

    def set(self, ary):
        if ary.shape != self.ioshape:
            raise ValueError('Invalid matrix shape')

        # Cast
        nary = np.asanyarray(ary, dtype=self.dtype, order='C')

        # Assign
        self.data[...,:nary.shape[-1]] = nary

    @property
    def _as_parameter_(self):
        # Return a pointer to the first element
        return self.data.ctypes.data

    @property
    def nbytes(self):
        return self.data.nbytes


class OpenMPMatrix(OpenMPMatrixBase, base.Matrix):
    def __init__(self, backend, ioshape, initval, tags):
        super(OpenMPMatrix, self).__init__(backend, backend.fpdtype, ioshape,
                                           initval, tags)


class OpenMPMatrixRSlice(base.MatrixRSlice):
    def __init__(self, backend, mat, p, q):
        super(OpenMPMatrixRSlice, self).__init__(backend, mat, p, q)

        # Since slices do not retain any information about the
        # high-order structure of an array it is fine to compact mat
        # down to two dimensions and simply slice this
        self.data = backend.compact_arr(mat.data)[p:q]

    @property
    def _as_parameter_(self):
        return self.data.ctypes.data


class OpenMPMatrixBank(base.MatrixBank):
    pass


class OpenMPConstMatrix(OpenMPMatrixBase, base.ConstMatrix):
    def __init__(self, backend, initval, tags):
        super(OpenMPConstMatrix, self).__init__(backend, backend.fpdtype,
                                                initval.shape, initval, tags)


class OpenMPMPIMatrix(OpenMPMatrix, base.MPIMatrix):
    pass


class OpenMPMPIView(base.MPIView):
    def __init__(self, backend, matmap, rcmap, stridemap, vlen, tags):
        super(OpenMPMPIView, self).__init__(backend, matmap, rcmap, stridemap,
                                            vlen, tags)


class OpenMPView(base.View):
    def __init__(self, backend, matmap, rcmap, stridemap, vlen, tags):
        super(OpenMPView, self).__init__(backend, matmap, rcmap, stridemap,
                                         vlen, tags)

        # Row/column indcies of each view element
        r, c = rcmap[...,0], rcmap[...,1]

        # We want to go from matrix objects and row/column indicies
        # to memory addresses.  The algorithm for this is:
        # ptr = m.base + r*m.pitch + c*itemsize
        ptrmap = np.array(c*self.refitemsize, dtype=np.intp)
        for m in self._mats:
            ix = np.where(matmap == m)
            ptrmap[ix] += m._as_parameter_ + r[ix]*m.pitch

        shape = (self.nrow, self.ncol)
        self.mapping = OpenMPMatrixBase(backend, np.intp, shape, ptrmap, tags)
        self.strides = OpenMPMatrixBase(backend, np.int32, shape, stridemap,
                                        tags)

    @property
    def nbytes(self):
        return self.mapping.nbytes + self.strides.nbytes


class OpenMPQueue(base.Queue):
    def __init__(self):
        # Last kernel we executed
        self._last = None

        # Active MPI requests
        self._mpireqs = []

        # Items waiting to be executed
        self._items = collections.deque()

    def __lshift__(self, items):
        self._items.extend(items)

    def __mod__(self, items):
        self.run()
        self << items
        self.run()

    def __nonzero__(self):
        return bool(self._items)

    def _exec_item(self, item, rtargs):
        if base.iscomputekernel(item):
            item.run(*rtargs)
        elif base.ismpikernel(item):
            item.run(self._mpireqs, *rtargs)
        else:
            raise ValueError('Non compute/MPI kernel in queue')
        self._last = item

    def _exec_next(self):
        item, rtargs = self._items.popleft()

        # If we are at a sequence point then wait for current items
        if self._at_sequence_point(item):
            self._wait()

        # Execute the item
        self._exec_item(item, rtargs)

    def _exec_nowait(self):
        while self._items and not self._at_sequence_point(self._items[0][0]):
            self._exec_item(*self._items.popleft())

    def _exec_nonblock(self):
        while self._items:
            kern = self._items[0][0]

            # See if kern will block
            if self._at_sequence_point(kern) or base.iscomputekernel(kern):
                break

            self._exec_item(*self._items.popleft())

    def _wait(self):
        if base.ismpikernel(self._last):
            MPI.Prequest.Waitall(self._mpireqs)
            self._mpireqs = []
        self._last = None

    def _at_sequence_point(self, item):
        if base.ismpikernel(self._last) and not base.ismpikernel(item):
            return True
        else:
            return False

    def run(self):
        while self._items:
            self._exec_next()
        self._wait()

    @staticmethod
    def runall(queues):
        # Fire off any non-blocking kernels
        for q in queues:
            q._exec_nonblock()

        while any(queues):
            # Execute a (potentially) blocking item from each queue
            for q in it.ifilter(None, queues):
                q._exec_nowait()

            # Now consider kernels which will wait
            for q in it.ifilter(None, queues):
                q._exec_next()
                q._exec_nonblock()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
