# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty

from collections import Sequence, defaultdict
from weakref import WeakSet

def recordalloc(type):
    def recordalloc_type(fn):
        def newfn(self, *args, **kwargs):
            rv = fn(self, *args, **kwargs)
            self._allocs[type].add(rv)
            return rv
        return newfn
    return recordalloc_type

class Backend(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._allocs = defaultdict(WeakSet)

    @recordalloc('data')
    def matrix(self, nrow, ncol, initval=None, tags=set()):
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
        return self._matrix(nrow, ncol, initval, tags)

    @abstractmethod
    def _matrix(self, nrow, ncol, initval, tags):
        pass

    @recordalloc('banks')
    def matrix_bank(self, mats, tags=set()):
        """Creates a bank of matrices from *mats*

        These matrices must be homogeneous.

        :param mats: Matrices to bank.
        :param tags: Implementation-specific metadata.

        :type mats: List of :class:`~pyfr.backends.base.Matrix`
        :rtype: :class:`~pyfr.backends.base.MatrixBank`
        """
        return self._matrix_bank(mats, tags)

    @abstractmethod
    def _matrix_bank(self, mats, tags):
        pass

    @recordalloc('data')
    def mpi_matrix(self, nrow, ncol, initval=None, tags=set()):
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
        return self._mpi_matrix(nrow, ncol, initval, tags)

    @abstractmethod
    def _mpi_matrix(self, nrow, ncol, initval, tags):
        pass

    @recordalloc('data')
    def const_matrix(self, initval, tags=set()):
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
        return self._const_matrix(initval, tags)

    @abstractmethod
    def _const_matrix(self, initval, tags):
        pass

    @recordalloc('data')
    def sparse_matrix(self, initval, tags=set()):
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
        return self._sparse_matrix(initval, tags)

    @abstractmethod
    def _sparse_matrix(self, initval, tags):
        pass

    def auto_const_sparse_matrix(self, initval, tags=set()):
        """Creates either a constant or sparse matrix from *initval*
        """
        if self._is_sparse(initval, tags):
            return self.sparse_matrix(initval, tags)
        else:
            return self.const_matrix(initval, tags)

    @abstractmethod
    def _is_sparse(self, mat, tags):
        """Determines if a *mat* is sparse or not
        """
        pass

    @recordalloc('data')
    def view(self, matmap, rcmap, tags=set()):
        """Uses mapping to create a view of mat

        :param matmap: Matrix of matrix objects.
        :param rcmap: Matrix of (row, column) indicies.
        :param tags: Implementation-specific metadata.

        :type mat: :class:`~pyfr.backends.base.Matrix`
        :type mapping: numpy.ndarray
        :type tags: set of str, optional
        :rtype: :class:`~pyfr.backends.base.View`
        """
        return self._view(matmap, rcmap, tags)

    @abstractmethod
    def _view(self, matmap, rcmap, tags):
        pass

    @recordalloc('data')
    def mpi_view(self, matmap, rcmap, tags=set()):
        """Creates a view whose contents can be exchanged using MPI"""
        return self._mpi_view(matmap, rcmap, tags)

    @abstractmethod
    def _mpi_view(self, matmap, rcmap, tags):
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
        return self._kernel(name, *args, **kwargs)

    @abstractmethod
    def _kernel(self, name, *args, **kwargs):
        pass

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


class _MatrixBase(object):
    @abstractmethod
    def get(self):
        """Contents as an *numpy.ndarray*"""
        pass

    @abstractproperty
    def nrow(self):
        """Number of rows"""
        pass

    @abstractproperty
    def ncol(self):
        """Number of columns"""
        pass

    @abstractproperty
    def nbytes(self):
        """Size in bytes"""
        pass


class Matrix(_MatrixBase):
    """Matrix abstract base class
    """
    @abstractmethod
    def set(self, ary):
        """Sets the contents of the matrix to be *ary*"""
        pass


class ConstMatrix(_MatrixBase):
    """Constant matrix abstract base class"""
    pass


class SparseMatrix(_MatrixBase):
    """Sparse matrix abstract base class"""
    pass


class MPIMatrix(Matrix):
    """MPI matrix abstract base class"""
    pass


class MatrixBank(_MatrixBase, Sequence):
    """Matrix bank abstract base class"""

    @abstractmethod
    def __init__(self, matrices):
        self._banks = matrices
        self._curr = self._banks[0]

    def __len__(self):
        return len(self._banks)

    def __getitem__(self, idx):
        return self._banks[idx]

    def __getattr__(self, attr):
        return getattr(self._curr, attr)

    def get(self):
        """Contents of the current bank as an *numpy.ndarray*"""
        return self._curr.get()

    def set(self, ary):
        """Sets the contents of the current bank to be *ary*"""
        self._curr.set(ary)

    @property
    def nrow(self):
        """Number of rows"""
        return self._curr.nrow

    @property
    def ncol(self):
        """Number of columns"""
        return self._curr.ncol

    def set_bank(self, idx):
        """Switches the currently active bank to *idx*"""
        self._curr = self._banks[idx]


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
