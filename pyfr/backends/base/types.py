# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import Sequence
 
import numpy as np


class MatrixBase(object):
    __metaclass__ = ABCMeta

    _base_tags = set()

    @abstractmethod
    def __init__(self, backend, ioshape, tags):
        self.backend = backend
        self.ioshape = ioshape
        self.tags = self._base_tags | tags

    @abstractmethod
    def get(self):
        pass

    @property
    def pitch(self):
        return self.leaddim*self.itemsize

    @property
    def traits(self):
        return (self.nrow, self.leaddim, self.leadsubdim, self.dtype)


class Matrix(MatrixBase):
    """Matrix abstract base class
    """
    _base_tags = {'dense'}

    @abstractmethod
    def set(self, buf):
        pass

    def rslice(self, p, q):
        return self.backend.matrix_rslice(self, p, q)


class MatrixRSlice(object):
    """Slice of a matrix abstract base class"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, backend, mat, p, q):
        self.backend = backend
        self.parent = mat

        if p < 0 or q > mat.nrow or q < p:
            raise ValueError('Invalid row slice')

        self.nrow, self.ncol = q - p, mat.ncol
        self.dtype, self.itemsize = mat.dtype, mat.itemsize
        self.leaddim, self.leadsubdim = mat.leaddim, mat.leadsubdim

        self.tags = mat.tags | {'slice'}

    @property
    def pitch(self):
        return self.leaddim*self.itemsize

    @property
    def traits(self):
        return (self.nrow, self.leaddim, self.leadsubdim, self.dtype)


class ConstMatrix(MatrixBase):
    """Constant matrix abstract base class"""
    _base_tags = {'const', 'dense'}


class MPIMatrix(Matrix):
    """MPI matrix abstract base class"""
    pass


class MatrixBank(Sequence):
    """Matrix bank abstract base class"""

    def __init__(self, backend, mats, initbank, tags):
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
    def __init__(self, backend, matmap, rcmap, stridemap, vlen, tags):
        self.nrow = nrow = matmap.shape[0]
        self.ncol = ncol = matmap.shape[1]
        self.vlen = vlen

        # Get the different matrices which we map onto
        self._mats = list(set(matmap.flat))

        # Extract the base allocation and data type
        self.basedata = self._mats[0].basedata
        self.refdtype = self._mats[0].dtype

        # For vector views a stridemap is required
        if vlen != 1 and np.any(stridemap == 0):
            raise ValueError('Vector views require a non-zero stride map')

        # Check all of the shapes match up
        if matmap.shape != rcmap.shape[:2] or matmap.shape != stridemap.shape:
            raise TypeError('Invalid view matrix shapes')

        # Validate the matrices
        if any(not isinstance(m, backend.matrix_cls) for m in self._mats):
            raise TypeError('Incompatible matrix type for view')

        if any(m.basedata != self.basedata for m in self._mats):
            raise TypeError('All viewed matrices must belong to the same '
                            'allocation extent')

        if any(m.dtype != self.refdtype for m in self._mats):
            raise TypeError('Mixed data types are not supported')


class MPIView(object):
    @abstractmethod
    def __init__(self, backend, matmap, rcmap, stridemap, vlen, tags):
        self.nrow = nrow = matmap.shape[0]
        self.ncol = ncol = matmap.shape[1]
        self.vlen = vlen

        # Create a normal view
        self.view = backend.view(matmap, rcmap, stridemap, vlen, tags)

        # Now create an MPI matrix so that the view contents may be packed
        self.mpimat = backend.mpi_matrix((nrow, vlen, ncol), tags=tags)


class Queue(object):
    """Kernel execution queue"""
    __metaclass__ = ABCMeta

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
