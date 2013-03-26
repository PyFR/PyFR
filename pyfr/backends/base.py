# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import Sequence, defaultdict
from functools import wraps
from inspect import getcallargs
from weakref import WeakSet

import numpy as np

from pyfr.util import ndrange, proxylist


def recordalloc(type):
    def recordalloc_type(fn):
        @wraps(fn)
        def newfn(self, *args, **kwargs):
            rv = fn(self, *args, **kwargs)
            self._allocs[type].add(rv)
            return rv
        return newfn
    return recordalloc_type


def traits(**tr):
    def traits_tr(fn):
        fn._traits = tr
        return fn
    return traits_tr


def issuitable(kern, *args, **kwargs):
    kernt = getattr(kern, '_traits', {})
    kargs = getcallargs(kern, *args, **kwargs)

    for k, tags in kernt.iteritems():
        argtags = kargs[k].tags
        for t in tags:
            if (t[0] == '!' and t[1:] in argtags) or\
               (t[0] != '!' and t not in argtags):
                return False

    return True


class Backend(object):
    __metaclass__ = ABCMeta

    # Backend name
    name = None

    @abstractmethod
    def __init__(self, cfg):
        assert self.name is not None

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
        return self.matrix_cls(self, ioshape, initval, iopacking, tags)

    @recordalloc('rslices')
    def matrix_rslice(self, mat, p, q):
        return self.matrix_rslice_cls(self, mat, p, q)

    @recordalloc('banks')
    def matrix_bank(self, mats, initbank=0, tags=set()):
        """Creates a bank of matrices from *mats*

        These matrices must be homogeneous.

        :param mats: Matrices to bank.
        :param tags: Implementation-specific metadata.

        :type mats: List of :class:`~pyfr.backends.base.Matrix`
        :rtype: :class:`~pyfr.backends.base.MatrixBank`
        """
        return self.matrix_bank_cls(self, mats, initbank, tags)

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
        return self.mpi_matrix_cls(self, ioshape, initval, iopacking, tags)

    def mpi_matrix_for_view(self, view, tags=set()):
        return self.mpi_matrix((view.nrow, view.ncol, view.vlen), tags=tags)

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
        return self.const_matrix_cls(self, initval, iopacking, tags)

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
        pass

    def block_diag_matrix(self, initval, brange, iopacking='AoS', tags=set()):
        return BlockDiagMatrix(self, initval, brange, iopacking, tags)

    def auto_matrix(self, initval, iopacking='AoS', tags=set()):
        """Creates either a constant or sparse matrix from *initval*
        """
        # HACK: The following code attempts to identify one special-
        # case of block diagonal matrices;  while it is currently
        # sufficient a more robust methodology is desirable.
        shape = initval.shape
        if iopacking == 'AoS' and len(shape) == 4 and shape[1] == shape[3]:
            for i, j in ndrange(shape[1], shape[1]):
                if i == j:
                    continue

                if np.any(initval[:,i,:,j] != 0):
                    break
            else:
                # Block spans are trivial
                brange = [(i*shape[0], (i + 1)*shape[0],
                           i*shape[2], (i + 1)*shape[2])
                          for i in xrange(shape[1])]

                return self.block_diag_matrix(initval, brange, iopacking,
                                              tags)

        # Not block-diagonal; return a constant matrix
        return self.const_matrix(initval, iopacking, tags)

    @recordalloc('view')
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
        return self.view_cls(self, matmap, rcmap, stridemap, vlen, tags)

    @recordalloc('view')
    def mpi_view(self, matmap, rcmap, stridemap=None, vlen=1, tags=set()):
        """Creates a view whose contents can be exchanged using MPI"""
        if stridemap is None:
            stridemap = np.ones(matmap.shape, dtype=np.int32)
        return self.mpi_view_cls(self, matmap, rcmap, stridemap, vlen, tags)

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
        for prov in self._providers:
            kern = getattr(prov, name, None)
            if kern and issuitable(kern, *args, **kwargs):
                return kern(*args, **kwargs)
        else:
            raise KeyError("'{}' has no providers".format(name))

    @recordalloc('queue')
    def queue(self):
        """Creates a queue

        :rtype: :class:`~pyfr.backends.base.Queue`
        """
        return self.queue_cls()

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
        self.queue_cls.runall(sequence)

    @property
    def nbytes(self):
        """Number of data bytes currently allocated on the backend"""
        return sum(d.nbytes for d in self._allocs['data'])

    @staticmethod
    def from_aos_stride_to_native(a, s):
        return 1, a

    @staticmethod
    def from_soa_stride_to_native(s, a):
        return a, 1

    @staticmethod
    def from_x_to_native(mat, cpacking):
        # Reorder if packed differently
        if cpacking != 'SoA':
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

    @staticmethod
    def from_native_to_x(mat, nshape, npacking):
        if npacking != 'SoA':
            n, nd = nshape, len(nshape)
            if nd == 3:
                mat = mat.reshape(n[0], n[2], n[1]).swapaxes(1, 2)
            elif nd == 4:
                mat = mat.reshape(n[1], n[0], n[3], n[2])\
                            .swapaxes(0, 1).swapaxes(2, 3)

        return mat.reshape(nshape)

    @staticmethod
    def _to_shape(shape, currpacking, newpacking):
        if currpacking not in ('AoS', 'SoA'):
            raise ValueError('Invalid matrix packing')

        if len(shape) == 2 or currpacking == newpacking:
            return shape
        elif len(shape) == 3:
            return shape[0], shape[2], shape[1]
        elif len(shape) == 4:
            return shape[1], shape[0], shape[3], shape[2]
        else:
            raise ValueError('Invalid matrix shape')

    @staticmethod
    def aos_shape(shape, packing):
        return Backend._to_shape(shape, packing, 'AoS')

    @staticmethod
    def soa_shape(shape, packing):
        return Backend._to_shape(shape, packing, 'SoA')

    @staticmethod
    def compact_shape(shape, packing, subcolmod=1):
        sshape = Backend.soa_shape(shape, packing)

        if len(sshape) == 2:
            return sshape[0], sshape[1], sshape[1]
        else:
            nrow = sshape[0] if len(sshape) == 3 else sshape[0]*sshape[1]

            # Align sub-columns
            lsd = sshape[-1] - (sshape[-1] % -subcolmod)
            ncol = sshape[-2]*lsd

            return nrow, ncol, lsd


class MatrixBase(object):
    __metaclass__ = ABCMeta

    _base_tags = set()

    @abstractmethod
    def __init__(self, backend, ioshape, iopacking, tags):
        self.backend = backend
        self.ioshape = ioshape
        self.iopacking = iopacking
        self.tags = self._base_tags | tags

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

    @property
    def aos_shape(self):
        return aos_shape(self.ioshape, self.iopacking)

    @property
    def soa_shape(self):
        return soa_shape(self.ioshape, self.iopacking)


class Matrix(MatrixBase):
    """Matrix abstract base class
    """
    _base_tags = {'dense'}

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
        self.tags = mat.tags | {'slice'}

    @property
    def nbytes(self):
        return 0


class ConstMatrix(MatrixBase):
    """Constant matrix abstract base class"""
    _base_tags = {'const', 'dense'}


class SparseMatrix(MatrixBase):
    """Sparse matrix abstract base class"""
    _base_tags = {'const', 'sparse'}


class BlockDiagMatrix(MatrixBase):
    _base_tags = {'const', 'blockdiag'}

    def __init__(self, backend, initval, brange, iopacking, tags):
        super(BlockDiagMatrix, self).__init__(backend, initval.shape,
                                              iopacking, tags)

        # Unpack into a dense matrix and extract the blocks
        self.matrix = matrix = self._pack(initval)
        self.blocks = [matrix[ri:rj,ci:cj] for ri, rj, ci, cj in brange]
        self.ranges = brange

    def _get(self):
        return self.matrix

    @property
    def nbytes(self):
        return 0


class MPIMatrix(Matrix):
    """MPI matrix abstract base class"""
    pass


class MatrixBank(Sequence):
    """Matrix bank abstract base class"""

    @abstractmethod
    def __init__(self, backend, matrices, initbank, tags):
        self.backend = backend
        self.tags = tags

        self._mats = matrices
        self._curr_idx = initbank
        self._curr_mat = self._mats[initbank]

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

    @property
    def nbytes(self):
        return sum(m.nbytes for m in self)


class View(object):
    """View abstract base class"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, backends, matmap, rcmap, stridemap, vlen, tags):
        self.nrow = nrow = matmap.shape[0]
        self.ncol = ncol = matmap.shape[1]
        self.vlen = vlen

        # For vector views a stridemap is required
        if vlen != 1 and np.any(stridemap == 0):
            raise ValueError('Vector views require a non-zero stride map')

        # Check all of the shapes match up
        if matmap.shape != rcmap.shape[:2] or\
           matmap.shape != stridemap.shape:
            raise TypeError('Invalid matrix shapes')

        # Get the different matrices which we map onto
        self._mats = list(np.unique(matmap))

    @abstractproperty
    def nbytes(self):
        pass


class MPIView(object):
    @abstractmethod
    def __init__(self, backend, matmap, rcmap, stridemap, vlen, tags):
        self.nrow = nrow = matmap.shape[0]
        self.ncol = ncol = matmap.shape[1]
        self.vlen = vlen

        # Create a normal view
        self.view = backend.view(matmap, rcmap, stridemap, vlen, tags)

        # Now create an MPI matrix so that the view contents may be packed
        self.mpimat = backend.mpi_matrix((nrow, ncol, vlen), None, 'AoS',
                                          tags=tags)

    @property
    def nbytes(self):
        return self.view.nbytes + self.mpimat.nbytes


class Kernel(object):
    """Bound kernel abstract base class"""
    __metaclass__ = ABCMeta

    def __call__(self, *args):
        return self, args

    @property
    def retval(self):
        return None

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class ComputeKernel(Kernel):
    pass


class MPIKernel(Kernel):
    pass


def iscomputekernel(kernel):
    return isinstance(kernel, ComputeKernel)


def ismpikernel(kernel):
    return isinstance(kernel, MPIKernel)


class _MetaKernel(object):
    def __init__(self, kernels):
        self._kernels = proxylist(kernels)

    def run(self, *args, **kwargs):
        self._kernels.run(*args, **kwargs)


class ComputeMetaKernel(_MetaKernel, ComputeKernel):
    pass


class MPIMetaKernel(_MetaKernel, MPIKernel):
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
