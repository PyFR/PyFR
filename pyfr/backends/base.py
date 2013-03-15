# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty

from collections import Sequence, defaultdict
from weakref import WeakSet

from functools import wraps

import numpy as np


def recordalloc(type):
    def recordalloc_type(fn):
        @wraps(fn)
        def newfn(self, *args, **kwargs):
            rv = fn(self, *args, **kwargs)
            self._allocs[type].add(rv)
            return rv
        return newfn
    return recordalloc_type


class Backend(object):
    __metaclass__ = ABCMeta

    # Backend name
    name = None

    # SoA or AoS
    packing = None

    def __init__(self, cfg):
        assert self.name is not None
        assert self.packing in ('SoA', 'AoS')

        self._cfg = cfg
        self._allocs = defaultdict(WeakSet)

    @recordalloc('data')
    def matrix(self, ioshape, initval=None, iopacking='AoS', tags=set()):
        """Creates an *nrow* by *ncol* matrix

        If an inital value is specified the shape of the provided
        numpy array must be (*nrow*, *ncol*).

        :param nrow: Number of rows.
        :param ncol: Number of columns.
        :param initval: Initial value of the matrix.
        :param tags: Implementation-specific metadata.

        :type nrow: int
        :type ncol: int
        :type initval: numpy.ndarray, optional
        :type tags: set of str, optional
        :rtype: :class:`~pyfr.backends.base.Matrix`
        """
        return self._matrix(ioshape, initval, iopacking, tags)

    @abstractmethod
    def _matrix(self, ioshape, initval, iopacking, tags):
        pass

    @recordalloc('rslices')
    def matrix_rslice(self, mat, p, q):
        return self._matrix_rslice(mat, p, q)

    @abstractmethod
    def _matrix_rslice(self, mat, p, q):
        pass

    @recordalloc('banks')
    def matrix_bank(self, mats, initbank=0, tags=set()):
        """Creates a bank of matrices from *mats*

        These matrices must be homogeneous.

        :param mats: Matrices to bank.
        :param tags: Implementation-specific metadata.

        :type mats: List of :class:`~pyfr.backends.base.Matrix`
        :rtype: :class:`~pyfr.backends.base.MatrixBank`
        """
        return self._matrix_bank(mats, initbank, tags)

    @abstractmethod
    def _matrix_bank(self, mats, tags):
        pass

    @recordalloc('data')
    def mpi_matrix(self, ioshape, initval=None, iopacking='AoS', tags=set()):
        """Creates a matrix which can be exchanged over MPI

        Since an MPI Matrix *is a* :class:`~pyfr.backends.base.Matrix`
        it can be used in *any* kernel which accepts a regular Matrix.

        :param nrow: Number of rows.
        :param ncol: Number of columns.
        :param initval: Initial value of the matrix.
        :param tags: Implementation-specific metadata.

        :type nrow: int
        :type ncol: int
        :type initval: numpy.ndarray, optional
        :type tags: set of str, optional
        :rtype: :class:`~pyfr.backends.base.MPIMatrix`
        """
        return self._mpi_matrix(ioshape, initval, iopacking, tags)

    def mpi_matrix_for_view(self, view, tags=set()):
        return self.mpi_matrix((view.nrow, view.ncol, view.vlen), tags=tags)

    @abstractmethod
    def _mpi_matrix(self, ioshape, iopacking, initval, tags):
        pass

    @recordalloc('data')
    def const_matrix(self, initval, iopacking='AoS', tags=set()):
        """Creates a constant matrix from *initval*

        This should be preferred over :meth:`matrix` when it is known
        at the point of instantiation that the resulting matrix will
        be invariant.  Backend implementations may use this for both
        parameter validation and kernel optimization.  It is expected
        that all kernels which accept a
        :class:`~pyfr.backends.base.Matrix` instance will also accept
        a constant matrix.

        :param initval: Initial value of the matrix.
        :param tags: Implementation-specific metadata.

        :type initval: numpy.ndarray
        :type tags: set of str, optional
        :rtype: :class:`~pyfr.backends.base.ConstMatrix`
        """
        return self._const_matrix(initval, iopacking, tags)

    @abstractmethod
    def _const_matrix(self, initval, tags):
        pass

    @recordalloc('data')
    def sparse_matrix(self, initval, iopacking='AoS', tags=set()):
        """Creates a sparse matrix from *initval*

        A *sparse matrix* is a special type of constant matrix
        that---potentially---uses a different storage pattern that is
        well-suited to matrices populated with a large number of zeros.

        :param initval: Initial value of the matrix.
        :param tags: Implementation-specific metadata.

        :type initval: numpy.ndarray
        :type tags: set of str, optional
        :rtype: :class:`~pyfr.backends.base.SparseMatrix`
        """
        return self._sparse_matrix(initval, iopacking, tags)

    @abstractmethod
    def _sparse_matrix(self, initval, iopacking, tags):
        pass

    def auto_const_sparse_matrix(self, initval, iopacking='AoS', tags=set()):
        """Creates either a constant or sparse matrix from *initval*
        """
        if self._is_sparse(initval, tags):
            return self.sparse_matrix(initval, iopacking, tags)
        else:
            return self.const_matrix(initval, iopacking, tags)

    @abstractmethod
    def _is_sparse(self, mat, tags):
        """Determines if a *mat* is sparse or not
        """
        pass

    @recordalloc('data')
    def view(self, matmap, rcmap, stridemap=None, vlen=1, tags=set()):
        """Uses mapping to create a view of mat

        :param matmap: Matrix of matrix objects.
        :param rcmap: Matrix of (row, column) indicies.
        :param tags: Implementation-specific metadata.

        :type mat: :class:`~pyfr.backends.base.Matrix`
        :type mapping: numpy.ndarray
        :type tags: set of str, optional
        :rtype: :class:`~pyfr.backends.base.View`
        """
        if stridemap is None:
            stridemap = np.ones(matmap.shape, dtype=np.int32)
        return self._view(matmap, rcmap, stridemap, vlen, tags)

    @abstractmethod
    def _view(self, matmap, rcmap, stridemap, vlen, tags):
        pass

    @recordalloc('data')
    def mpi_view(self, matmap, rcmap, stridemap=None, vlen=1, tags=set()):
        """Creates a view whose contents can be exchanged using MPI"""
        if stridemap is None:
            stridemap = np.ones(matmap.shape, dtype=np.int32)
        return self._mpi_view(matmap, rcmap, stridemap, vlen, tags)

    @abstractmethod
    def _mpi_view(self, matmap, rcmap, stridemap, vlen, tags):
        pass

    @recordalloc('kern')
    def kernel(self, name, *args, **kwargs):
        """Locates and binds a kernel called *name*

        Searches for a kernel called *name* and---if found---attempts
        to bind it with the provided arguments and keyword arguments.
        It is possible that a backend may have multiple implementations
        of a given kernel.  In such an instance it is expected that a
        given backend will exploit the specific type and *tags* of
        any relevant arguments in order to yield an optimal
        implementation.

        :rtype: :class:`~pyfr.backends.base.Kernel`
        """
        for prov in reversed(self._providers):
            try:
                kern = getattr(prov, name)
            except AttributeError:
                continue

            return kern(*args, **kwargs)
        else:
            raise KeyError("'{}' has no providers".format(name))

    @recordalloc('queue')
    def queue(self):
        """Creates a queue

        :rtype: :class:`~pyfr.backends.base.Queue`
        """
        return self._queue()

    @abstractmethod
    def _queue(self):
        pass

    @abstractmethod
    def runall(self, sequence):
        """Executes all of the kernels in the provided sequence of queues

        Given a sequence of :class:`~pyfr.backends.base.Queue` instances
        this method runs all of the kernels in the queues in an
        efficent manner.  This is done under the assumption that the
        kernels in one queue are independent from those in another.
        It is, however, guarenteed that kernels *inside* of a queue
        will be executed in the order in which they were added to the
        queue.
        """
        pass

    @property
    def nbytes(self):
        """Number of data bytes currently allocated on the backend"""
        return sum(d.nbytes for d in self._allocs['data'])

    def from_aos_stride_to_native(self, a, s):
        return (s, 1) if self.packing == 'AoS' else (1, a)

    def from_soa_stride_to_native(self, s, a):
        return (a, 1) if self.packing == 'SoA' else (1, s)

    def from_x_to_native(self, mat, cpacking):
        # Reorder if packed differently
        if self.packing != cpacking:
            if mat.ndim == 3:
                mat = mat.swapaxes(1, 2)
            elif mat.ndim == 4:
                mat = mat.swapaxes(0, 1).swapaxes(2, 3)

        # Compact down to two dimensions
        if mat.ndim == 2:
            return mat
        elif mat.ndim == 3:
            return mat.reshape(mat.shape[0], -1)
        elif mat.ndim == 4:
            return mat.reshape(mat.shape[0]*mat.shape[1], -1)

    def from_native_to_x(self, mat, nshape, npacking):
        if self.packing != npacking:
            n, nd = nshape, len(nshape)
            if nd == 3:
                mat = mat.reshape(n[0], n[2], n[1]).swapaxes(1, 2)
            elif nd == 4:
                mat = mat.reshape(n[1], n[0], n[3], n[2])\
                         .swapaxes(0, 1).swapaxes(2, 3)

        return mat.reshape(nshape)


class MatrixBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, backend, ioshape, iopacking, tags):
        self.backend = backend
        self.ioshape = ioshape
        self.iopacking = iopacking
        self.tags = tags

        if len(ioshape) == 2:
            self.nrow = ioshape[0]
            self.ncol = ioshape[1]
        elif len(ioshape) == 3:
            self.nrow = ioshape[0]
            self.ncol = ioshape[1]*ioshape[2]
        elif len(ioshape) == 4:
            self.nrow = ioshape[0]*ioshape[1]
            self.ncol = ioshape[2]*ioshape[3]
        else:
            raise ValueError('Invalid matrix I/O shape')

    def _pack(self, buf):
        if buf.shape != self.ioshape:
            raise ValueError('Invalid matrix shape')

        return self.backend.from_x_to_native(buf, self.iopacking)

    def _unpack(self, buf):
        if buf.shape != (self.nrow, self.ncol):
            raise ValueError('Invalid matrix shape')

        return self.backend.from_native_to_x(buf, self.ioshape, self.iopacking)

    def get(self):
        return self._unpack(self._get())

    @abstractmethod
    def _get(self):
        pass

    @abstractproperty
    def nbytes(self):
        """Size in bytes"""
        pass


class Matrix(MatrixBase):
    """Matrix abstract base class
    """
    def set(self, buf):
        return self._set(self._pack(buf))

    def rslice(self, p, q):
        return self.backend.matrix_rslice(self, p, q)

    @abstractmethod
    def _set(self, buf):
        pass


class MatrixRSlice(object):
    """Slice of a matrix abstract base class"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, backend, mat, p, q):
        self.backend = backend
        self.parent = mat

        if p < 0 or q > mat.nrow or q < p:
            raise ValueError('Invalid row slice')

        self.nrow = q - p
        self.ncol = mat.ncol
        self.tags = mat.tags


class ConstMatrix(MatrixBase):
    """Constant matrix abstract base class"""
    pass


class SparseMatrix(MatrixBase):
    """Sparse matrix abstract base class"""
    pass


class MPIMatrix(Matrix):
    """MPI matrix abstract base class"""
    pass


class MatrixBank(Sequence):
    """Matrix bank abstract base class"""

    @abstractmethod
    def __init__(self, matrices, initbank, tags):
        self._mats = matrices

        self._curr_idx = initbank
        self._curr_mat = self._mats[initbank]

    def __len__(self):
        return len(self._mats)

    def __getitem__(self, idx):
        return self._mats[idx]

    def __getattr__(self, attr):
        return getattr(self._curr_mat, attr)

    @property
    def active(self):
        return self._curr_idx

    @active.setter
    def active(self, idx):
        self._curr_idx = idx
        self._curr_mat = self._mats[idx]

    @property
    def nbytes(self):
        return sum(m.nbytes for m in self)


class View(object):
    """View abstract base class"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def nbytes(self):
        pass


class MPIView(View):
    """MPI view abstract base class"""
    pass


class Kernel(object):
    """Bound kernel abstract base class"""
    __metaclass__ = ABCMeta

    def __call__(self, *args):
        return self, args

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


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
