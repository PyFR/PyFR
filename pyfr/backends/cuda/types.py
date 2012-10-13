# -*- coding: utf-8 -*-

import pycuda.driver as cuda
import numpy as np

import pyfr.backends.base as base

from pyfr.backends.cuda.util import memcpy2d_htod, memcpy2d_dtoh

class _CudaBase2D(object):
    def __init__(self, backend, nrow, ncol, initval=None, tags={}):
        self._nrow = nrow
        self._ncol = ncol
        self.tags = tags

        # Compute the size, in bytes, of the minor dimension
        mindimsz = self.mindim*self.itemsize

        if 'nopad' not in tags:
            # Allocate a 2D array aligned to the major dimension
            self.data, self.pitch = cuda.mem_alloc_pitch(mindimsz, self.majdim,
                                                         self.itemsize)

            # Ensure that the pitch is a multiple of itemsize
            assert (self.pitch % self.itemsize) == 0
        else:
            # Allocate a standard, tighly packed, array
            self.data = cuda.mem_alloc(mindimsz*self.majdim)
            self.pitch = mindimsz

        # Process any initial values
        if initval is not None:
            # Check the sizes match up
            if initval.shape != (nrow, ncol):
                raise ValueError('Initial matrix has invalid dimensions')

            # Convert
            nary = np.asanyarray(initval, dtype=self.dtype, order=self.order)

            # Copy
            memcpy2d_htod(self.data, nary, mindimsz, self.pitch, mindimsz,
                          self.majdim)

    def get(self):
        # Allocate an empty buffer
        buf = np.empty((self.nrow, self.ncol), dtype=self.dtype,
                       order=self.order)

        # Copy
        memcpy2d_dtoh(buf, self.data, self.pitch, self.mindim*self.itemsize,
                      self.mindim*self.itemsize, self.majdim)

        return buf

    def offsetof(self, i, j):
        if i >= self._nrow or j >= self._ncol:
            raise ValueError('Index ({},{}) out of bounds ({},{}))'.\
                             format(i, j, self._nrow, self._ncol))

        return self.pitch*i + j*self.itemsize if self.order == 'C' else\
               self.pitch*j + i*self.itemsize

    @property
    def nrow(self):
        return self._nrow

    @property
    def ncol(self):
        return self._ncol

    @property
    def itemsize(self):
        return np.dtype(self.dtype).itemsize

    @property
    def majdim(self):
        return self._nrow if self.order == 'C' else self._ncol

    @property
    def mindim(self):
        return self._ncol if self.order == 'C' else self._nrow

    @property
    def leaddim(self):
        return self.pitch / self.itemsize


class CudaMatrix(_CudaBase2D, base.Matrix):
    order = 'F'
    dtype = np.float64


class CudaMatrixBank(base.MatrixBank):
    def __init__(self, backend, nrow, ncol, nbanks, initval=None, tags={}):
        banks = [backend.matrix(nrow, ncol, initval, tags)
                 for i in xrange(0, nbanks)]
        super(CudaMatrixBank, self).__init__(banks)


class CudaConstMatrix(CudaMatrix, base.ConstMatrix):
    def __init__(self, backend, initval, tags={}):
        nrow, ncol = initval.shape
        return super(CudaMatrix, self).__init__(backend, nrow, ncol,
                                                initval, tags)


class CudaSparseMatrix(object):
    def __init__(self, backend, initval, tags={}):
        raise NotImplementedError('SparseMatrix todo!')


class CudaView(_CudaBase2D, base.View):
    order = 'F'
    dtype = np.intp

    def __init__(self, backend, mapping, tags={}):
        nrow, ncol = mapping.shape[:2]

        # Get the different matrices which we map onto
        self._mats = list(np.unique(mapping[:,:,0]))

        print len(self._mats)

        # Extract the data type and item size from the first matrix
        self.refdtype = self._mats[0].dtype
        self.refitemsize = self._mats[0].itemsize

        # Validate the matrices
        for m in self._mats:
            if not isinstance(m, CudaMatrix):
                raise TypeError('Incompatible matrix type for view')

            if m.dtype != self.refdtype:
                raise TypeError('Mixed view matrix types are not supported')


        # Convert the (mat, row, col) triplet to a device pointer
        mappingptr = np.apply_along_axis(lambda x: self._toptr(*x), 2, mapping)

        # Create a matrix on the GPU of these pointers
        super(CudaView, self).__init__(backend, nrow, ncol, mappingptr, tags)

    @staticmethod
    def _toptr(mat, i, j):
        return np.intp(int(mat.data) + mat.offsetof(i, j))



class CudaMPIView(CudaView, base.MPIView):
    def __init__(self, backend, mapping, tags={}):
        # Call the standard view constructor
        super(CudaMPIView, self).__init__(backend, mapping, tags)

        # Allocate a packing buffer on the GPU
        self.gbuf = cuda.mem_alloc(self.nrow*self.ncol*self.refitemsize)

        # Allocate an equiv page-locked buffer on the host for copying
        self.hbuf = cuda.pagelocked_empty((self.nrow, self.ncol),
                                          self.refdtype, self.order)
