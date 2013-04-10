# -*- coding: utf-8 -*-

import collections
import itertools as it

from mpi4py import MPI
import numpy as np
import pycuda.driver as cuda

import pyfr.backends.base as base
from pyfr.backends.cuda.util import memcpy2d_htod, memcpy2d_dtoh


class CudaMatrixBase(base.MatrixBase):
    def __init__(self, backend, dtype, ioshape, initval, iopacking, tags):
        super(CudaMatrixBase, self).__init__(backend, ioshape, iopacking, tags)

        # Data type info
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize

        # Dimensions
        nrow, ncol = backend.compact_shape(ioshape, iopacking)
        self.nrow = nrow
        self.ncol = ncol

        # Compute the size, in bytes, of the minor dimension
        colsz = self.ncol*self.itemsize

        if 'align' in tags:
            # Allocate a 2D array aligned to the major dimension
            self.data, self.pitch = cuda.mem_alloc_pitch(colsz, nrow,
                                                         self.itemsize)
            self._nbytes = nrow*self.pitch

            # Ensure that the pitch is a multiple of itemsize
            assert (self.pitch % self.itemsize) == 0
        else:
            # Allocate a standard, tighly packed, array
            self._nbytes = colsz*nrow
            self.data = cuda.mem_alloc(self._nbytes)
            self.pitch = colsz

        self.leaddim = self.pitch / self.itemsize
        self.leadsubdim = self.soa_shape[-1]
        self.traits = (nrow, self.leaddim, self.leadsubdim, self.dtype)

        # Zero the entire matrix (incl. slack)
        assert (self._nbytes % 4) == 0
        cuda.memset_d32(self.data, 0, self._nbytes/4)

        # Process any initial values
        if initval is not None:
            self.set(initval)

    def get(self):
        # Allocate an empty buffer
        buf = np.empty((self.nrow, self.ncol), dtype=self.dtype)

        # Copy
        memcpy2d_dtoh(buf, self.data, self.pitch, self.ncol*self.itemsize,
                      self.ncol*self.itemsize, self.nrow)

        # Reshape from a matrix to an SoA-packed array
        buf = buf.reshape(self.soa_shape)

        # Repack if the IO format is AoS
        if self.iopacking == 'AoS':
            buf = self.backend.aos_arr(buf, 'SoA')

        return buf

    def set(self, ary):
        if ary.shape != self.ioshape:
            raise ValueError('Invalid matrix shape')

        # Cast and repack into the SoA format
        nary = np.asanyarray(ary, dtype=self.dtype, order='C')
        nary = self.backend.compact_arr(nary, self.iopacking)

        # Copy
        memcpy2d_htod(self.data, nary, self.ncol*self.itemsize, self.pitch,
                      self.ncol*self.itemsize, self.nrow)

    @property
    def _as_parameter_(self):
        return long(self.data)

    def __long__(self):
        return long(self.data)

    @property
    def nbytes(self):
        return self._nbytes


class CudaMatrix(CudaMatrixBase, base.Matrix):
    def __init__(self, backend, ioshape, initval, iopacking, tags):
        super(CudaMatrix, self).__init__(backend, backend.fpdtype, ioshape,
                                         initval, iopacking, tags)


class CudaMatrixRSlice(base.MatrixRSlice):
    def __init__(self, backend, mat, p, q):
        super(CudaMatrixRSlice, self).__init__(backend, mat, p, q)

        # Copy over common attributes
        self.dtype, self.itemsize = mat.dtype, mat.itemsize
        self.pitch, self.leaddim = mat.pitch, mat.leaddim

        # Traits are those of a thinner matrix
        self.traits = (self.nrow, self.leaddim, self.dtype)

        # Starting offset of our row
        self._soffset = p*mat.pitch

    @property
    def _as_parameter_(self):
        return long(self.parent) + self._soffset

    @property
    def __long__(self):
        return long(self.parent) + self._soffset


class CudaMatrixBank(base.MatrixBank):
    def __init__(self, backend, mats, initbank, tags):
        if any(m.traits != mats[0].traits for m in mats[1:]):
            raise ValueError('Matrices in a bank must be homogeneous')

        super(CudaMatrixBank, self).__init__(backend, mats, initbank, tags)

    def __long__(self):
        return long(self._curr_mat)


class CudaConstMatrix(CudaMatrixBase, base.ConstMatrix):
    def __init__(self, backend, initval, iopacking, tags):
        ioshape = initval.shape
        super(CudaConstMatrix, self).__init__(backend, backend.fpdtype,
                                              ioshape, initval, iopacking,
                                              tags)


class CudaView(base.View):
    def __init__(self, backend, matmap, rcmap, stridemap, vlen, tags):
        super(CudaView, self).__init__(backend, matmap, rcmap, stridemap,
                                       vlen, tags)

        # Extract the data type and item size from the first matrix
        self.refdtype = self._mats[0].dtype
        self.refitemsize = self._mats[0].itemsize

        # Validate the matrices
        for m in self._mats:
            if not isinstance(m, (CudaMatrix, CudaConstMatrix)):
                raise TypeError('Incompatible matrix type for view')

            if m.dtype != self.refdtype:
                raise TypeError('Mixed view matrix types are not supported')

        # Row/column indcies of each view element
        r, c = rcmap[...,0], rcmap[...,1]

        # We want to go from matrix objects and row/column indicies
        # to memory addresses.  The algorithm for this is:
        # ptr = m.base + r*m.pitch + c*itemsize
        ptrmap = np.array(c*self.refitemsize, dtype=np.intp)
        for m in self._mats:
            ix = np.where(matmap == m)
            ptrmap[ix] += long(m) + r[ix]*m.pitch

        shape = (self.nrow, self.ncol)
        self.mapping = CudaMatrixBase(backend, np.intp, shape, ptrmap, 'AoS',
                                      tags)
        self.strides = CudaMatrixBase(backend, np.int32, shape, stridemap,
                                      'AoS', tags)

    @property
    def nbytes(self):
        return self.mapping.nbytes + self.strides.nbytes


class CudaMPIMatrix(CudaMatrix, base.MPIMatrix):
    def __init__(self, backend, ioshape, initval, iopacking, tags):
        # Call the standard matrix constructor
        super(CudaMPIMatrix, self).__init__(backend, ioshape, initval,
                                            iopacking, tags)

        # Allocate a page-locked buffer on the host for MPI to send/recv from
        self.hdata = cuda.pagelocked_empty((self.nrow, self.ncol),
                                           self.dtype, 'C')


class CudaMPIView(base.MPIView):
    def __init__(self, backend, matmap, rcmap, stridemap, vlen, tags):
        super(CudaMPIView, self).__init__(backend, matmap, rcmap, stridemap,
                                          vlen, tags)


class CudaQueue(base.Queue):
    def __init__(self):
        # Last kernel we executed
        self._last = None

        # CUDA stream and MPI request list
        self._stream_comp = cuda.Stream()
        self._stream_copy = cuda.Stream()
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
            item.run(self._stream_comp, self._stream_copy, *rtargs)
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

    def _wait(self):
        if base.iscomputekernel(self._last):
            self._stream_comp.synchronize()
            self._stream_copy.synchronize()
        elif base.ismpikernel(self._last):
            MPI.Prequest.Waitall(self._mpireqs)
            self._mpireqs = []
        self._last = None

    def _at_sequence_point(self, item):
        iscompute, ismpi = base.iscomputekernel, base.ismpikernel

        if (iscompute(self._last) and not iscompute(item)) or\
           (ismpi(self._last) and not ismpi(item)):
            return True
        else:
            return False

    def run(self):
        while self._items:
            self._exec_next()
        self._wait()

    @staticmethod
    def runall(queues):
        # First run any items which will not result in an implicit wait
        for q in queues:
            q._exec_nowait()

        # So long as there are items remaining in the queues
        while any(queues):
            # Execute a (potentially) blocking item from each queue
            for q in it.ifilter(None, queues):
                q._exec_next()
                q._exec_nowait()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
