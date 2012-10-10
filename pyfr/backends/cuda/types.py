# -*- coding: utf-8 -*-

import pycuda.driver as cuda
import numpy as np

from pyfr.backends.cuda.util import memcpy2d_htod, memcpy2d_dtoh

class _CudaBase2D(object):
    def __init__(self, backend, nrow, ncol, initval=None, tags={}):
        self.nrow = nrow
        self.ncol = ncol
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

    @property
    def itemsize(self):
        return np.dtype(self.dtype).itemsize

    @property
    def majdim(self):
        return self.nrow if self.order == 'C' else self.ncol

    @property
    def mindim(self):
        return self.ncol if self.order == 'C' else self.nrow

    @property
    def leaddim(self):
        return self.pitch / self.itemsize


class CudaMatrix(_CudaBase2D):
    order = 'F'
    dtype = np.float64


class CudaConstMatrix(CudaMatrix):
    pass


class CudaSparseMatrix(object):
    pass


class CudaView(_CudaBase2D):
    order = 'F'
    dtype = np.uint32

    def __init__(self, backend, mat, mapping, tags={}):
        nrow, ncol = mapping.shape[:2]
        self.viewof = mat

        # First validate the row and column numbers in mapping
        if mapping[:,:,0].max() >= mat.nrow or\
           mapping[:,:,1].max() >= mat.ncol:
            raise ValueError('Bad row/column numbers in view mapping')

        # Next get a vector to convert from (row,col) to an offset
        dvect = [mat.leaddim, 1] if mat.order == 'C' else [1, mat.leaddim]

        # Dot this with initval to give an nrow*ncol matrix of offsets
        mappingoff = mapping.dot(dvect)

        super(CudaView, self).__init__(backend, nrow, ncol, mappingoff, tags)


class CudaHostView(CudaView):
    def __init__(self, backend, mat, mapping, tags={}):
        # Call the standard view constructor
        super(CudaHostView, self).__init__(backend, mat, mapping, tags)

        # Allocate a packing buffer on the GPU
        self.gbuf = cuda.mem_alloc(self.nrow*self.ncol*mat.itemsize)

        # Allocate an equiv page-locked buffer on the host for copying
        self.hbuf = cuda.pagelocked_empty((self.nrow, self.ncol),
                                          mat.dtype, self.order)
