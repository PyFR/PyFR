# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import Sequence
 
import numpy as np


class MatrixBase(object):
    __metaclass__ = ABCMeta

    _base_tags = set()

    def __init__(self, backend, dtype, ioshape, initval, extent, tags):
        self.backend = backend
        self.tags = self._base_tags | tags

        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize

        # Alignment requirement for the leading dimension
        ldmod = backend.alignb // self.itemsize if 'align' in tags else 1

        # Our shape and dimensionality
        shape, ndim = list(ioshape), len(ioshape)

        if ndim == 2:
            nrow, ncol = shape
        elif ndim == 3 or ndim == 4:
            nrow = shape[0] if ndim == 3 else shape[0]*shape[1]
            ncol = shape[-2]*shape[-1] + (1 - shape[-2])*(shape[-1] % -ldmod)

        # Pad the final dimension
        shape[-1] -= shape[-1] % -ldmod

        # Assign
        self.nrow, self.ncol = nrow, ncol
        self.ioshape, self.datashape = ioshape, shape

        self.leaddim = ncol - (ncol % -ldmod)
        self.leadsubdim = shape[-1]

        self.pitch = self.leaddim*self.itemsize
        self.traits = (self.nrow, self.leaddim, self.leadsubdim, self.dtype)

        # Allocate
        backend.malloc(self, nrow*self.leaddim*self.itemsize, extent)

        # Process the initial value
        if initval is not None:
            if initval.shape != self.ioshape:
                raise ValueError('Invalid initial value')

            self._initval = np.asanyarray(initval, dtype=self.dtype)
        else:
            self._initval = None

    def get(self):
        # If we are yet to be allocated use our initial value
        if hasattr(self, '_initval'):
            if self._initval is not None:
                return self._initval
            else:
                return np.zeros(self.ioshape, dtype=self.dtype)
        # Otherwise defer to the backend
        else:
            return self._get()

    @abstractmethod
    def _get(self):
        pass


class Matrix(MatrixBase):
    """Matrix abstract base class
    """
    _base_tags = {'dense'}

    def set(self, ary):
        if ary.shape != self.ioshape:
            raise ValueError('Invalid matrix shape')

        # If we are yet to be allocated then update our initial value
        if hasattr(self, '_initval'):
            self._initval = np.asanyarray(ary, dtype=self.dtype)
        # Otherwise defer to the backend
        else:
            self._set(ary)

    @abstractmethod
    def _set(self, ary):
        pass

    def rslice(self, p, q):
        return self.backend.matrix_rslice(self, p, q)


class MatrixRSlice(object):
    def __init__(self, backend, mat, p, q):
        self.backend = backend
        self.parent = mat

        if p < 0 or q > mat.nrow or q < p:
            raise ValueError('Invalid row slice')

        self.p, self.q = p, q
        self.nrow, self.ncol = q - p, mat.ncol
        self.dtype, self.itemsize = mat.dtype, mat.itemsize
        self.leaddim, self.leadsubdim = mat.leaddim, mat.leadsubdim

        self.pitch = self.leaddim*self.itemsize
        self.traits = (self.nrow, self.leaddim, self.leadsubdim, self.dtype)

        self.tags = mat.tags | {'slice'}

    @property
    def basedata(self):
        return self.parent.basedata

    @property
    def offset(self):
        return self.parent.offset + self.p*self.pitch


class ConstMatrix(MatrixBase):
    """Constant matrix abstract base class"""
    _base_tags = {'const', 'dense'}


class MPIMatrix(Matrix):
    """MPI matrix abstract base class"""
    pass


class MatrixBank(Sequence):
    """Matrix bank abstract base class"""

    def __init__(self, backend, mats, initbank, tags):
        mats = list(mats)

        # Ensure all matrices have the same traits
        if any(m.traits != mats[0].traits for m in mats[1:]):
            raise ValueError('Matrices in a bank must be homogeneous')

        # Check that all matrices share tags
        if any(m.tags != mats[0].tags for m in mats[1:]):
            raise ValueError('Matrices in a bank must share tags')

        self.backend = backend
        self.tags = tags | mats[0].tags

        self._mats = mats
        self._curr_idx = initbank
        self._curr_mat = mats[initbank]

    def __len__(self):
        return len(self._mats)

    def __getitem__(self, idx):
        return self._mats[idx]

    def __getattr__(self, attr):
        return getattr(self._curr_mat, attr)

    def rslice(self, p, q):
        raise RuntimeError('Matrix banks can not be sliced')

    @property
    def active(self):
        return self._curr_idx

    @active.setter
    def active(self, idx):
        self._curr_idx = idx
        self._curr_mat = self._mats[idx]


class View(object):
    """View abstract base class"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, backend, matmap, rcmap, stridemap, vshape, tags):
        self.n = len(matmap)
        self.nvrow = vshape[-2] if len(vshape) == 2 else 1
        self.nvcol = vshape[-1] if len(vshape) >= 1 else 1
        self.rstrides = self.cstrides = None

        # Get the different matrices which we map onto
        self._mats = list(set(matmap.flat))

        # Extract the base allocation and data type
        self.basedata = self._mats[0].basedata
        self.refdtype = self._mats[0].dtype

        # Valid matrix types
        mattypes = (backend.matrix_cls, backend.matrix_rslice_cls)

        # Validate the matrices
        if any(not isinstance(m, mattypes) for m in self._mats):
            raise TypeError('Incompatible matrix type for view')

        if any(m.basedata != self.basedata for m in self._mats):
            raise TypeError('All viewed matrices must belong to the same '
                            'allocation extent')

        if any(m.dtype != self.refdtype for m in self._mats):
            raise TypeError('Mixed data types are not supported')

        # Base offsets and leading dimensions for each point
        offset = np.empty(self.n, dtype=np.int32)
        leaddim = np.empty(self.n, dtype=np.int32)

        for m in self._mats:
            ix = np.where(matmap == m)
            offset[ix], leaddim[ix] = m.offset // m.itemsize, m.leaddim

        # Go from matrices + row/column indcies to displacements
        # relative to the base allocation address
        self.mapping = (offset + rcmap[:,0]*leaddim + rcmap[:,1])[None,:]

        # Row strides
        if self.nvrow > 1:
            self.rstrides = (stridemap[:,0]*leaddim)[None,:]

        # Column strides
        if self.nvcol > 1:
            self.cstrides = stridemap[:,-1][None,:]


class MPIView(object):
    def __init__(self, backend, matmap, rcmap, stridemap, vshape, tags):
        # Create a normal view
        self.view = backend.view(matmap, rcmap, stridemap, vshape, tags)

        # Dimensions
        self.n = n = self.view.n
        self.nvrow = nvrow = self.view.nvrow
        self.nvcol = nvcol = self.view.nvcol

        # Now create an MPI matrix so that the view contents may be packed
        self.mpimat = backend.mpi_matrix((nvrow, nvcol, n), tags=tags)


class Queue(object):
    """Kernel execution queue"""
    __metaclass__ = ABCMeta

    def __init__(self, backend):
        self.backend = backend

    @abstractmethod
    def __lshift__(self, iterable):
        """Appends the kernels in *iterable* to the queue

        .. note::
          This method does **not** execute the kernels, but rather just
          schedules them.  Queued kernels should be executed by
          calling :meth:`pyfr.backends.base.Backend.runall`
        """
        pass

    @abstractmethod
    def __mod__(self, iterable):
        """Synchronously executes the kernels in *iterable*

        .. note::
          In the (unusual) instance that the queue already has one or
          more kernels queued these will run first.
        """
        pass
