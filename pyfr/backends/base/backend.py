# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from inspect import getcallargs
from weakref import WeakKeyDictionary

import numpy as np

from pyfr.util import ndrange


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


class BaseBackend(object):
    __metaclass__ = ABCMeta

    # Backend name
    name = None

    @abstractmethod
    def __init__(self, cfg):
        assert self.name is not None

        self.cfg = cfg

        # Numeric data type
        prec = cfg.get('backend', 'precision', 'double')
        if prec not in {'single', 'double'}:
            raise ValueError('Backend precision must be either single or '
                             'double')

        # Convert to a NumPy data type
        self.fpdtype = np.dtype(prec).type

        # Pending and committed allocation extents
        self._pend_extents = defaultdict(list)
        self._comm_extents = set()

        # Mapping from backend objects to memory extents
        self._obj_extents = WeakKeyDictionary()

    def malloc(self, obj, nbytes, extent):
        # If no extent has been specified then use a dummy object
        extent = extent if extent is not None else object()

        # Check that the extent has not already been committed
        if extent in self._comm_extents:
            raise ValueError('Extent "{}" has already been allocated'
                             .format(extent))

        # Append
        self._pend_extents[extent].append((obj, nbytes))

    def commit(self):
        for reqs in self._pend_extents.itervalues():
            # Determine the required allocation size
            sz = sum(nbytes - (nbytes % -self.alignb) for _, nbytes in reqs)

            # Perform the allocation
            data = self._malloc_impl(sz)

            offset = 0
            for obj, nbytes in reqs:
                # Fire the objects allocation callback
                obj.onalloc(data, offset)

                # Increment the offset
                offset += nbytes - (nbytes % -self.alignb)

                # Retain a (weak) reference to the allocated extent
                self._obj_extents[obj] = data

        # Mark the extents as committed and clear
        self._comm_extents.update(self._pend_extents)
        self._pend_extents.clear()

    @abstractmethod
    def _malloc_impl(self, nbytes):
        pass

    def matrix(self, ioshape, initval=None, extent=None, tags=set()):
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
        return self.matrix_cls(self, ioshape, initval, extent, tags)

    def matrix_rslice(self, mat, p, q):
        return self.matrix_rslice_cls(self, mat, p, q)

    def matrix_bank(self, mats, initbank=0, tags=set()):
        """Creates a bank of matrices from *mats*

        These matrices must be homogeneous.

        :param mats: Matrices to bank.
        :param tags: Implementation-specific metadata.

        :type mats: List of :class:`~pyfr.backends.base.Matrix`
        :rtype: :class:`~pyfr.backends.base.MatrixBank`
        """
        return self.matrix_bank_cls(self, mats, initbank, tags)

    def mpi_matrix(self, ioshape, initval=None, extent=None, tags=set()):
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
        return self.mpi_matrix_cls(self, ioshape, initval, extent, tags)

    def mpi_matrix_for_view(self, view, tags=set()):
        return self.mpi_matrix((view.nvrow, view.nvcol, view.n), tags=tags)

    def const_matrix(self, initval, extent=None, tags=set()):
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
        return self.const_matrix_cls(self, initval, extent, tags)

    def view(self, matmap, rcmap, stridemap=None, vshape=tuple(), tags=set()):
        """Uses mapping to create a view of mat

        :param matmap: Matrix of matrix objects.
        :param rcmap: Matrix of (row, column) indicies.
        :param tags: Implementation-specific metadata.

        :type mat: :class:`~pyfr.backends.base.Matrix`
        :type mapping: numpy.ndarray
        :type tags: set of str, optional
        :rtype: :class:`~pyfr.backends.base.View`
        """
        return self.view_cls(self, matmap, rcmap, stridemap, vshape, tags)

    def mpi_view(self, matmap, rcmap, stridemap=None, vshape=tuple(),
                 tags=set()):
        """Creates a view whose contents can be exchanged using MPI"""
        return self.mpi_view_cls(self, matmap, rcmap, stridemap, vshape, tags)

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

    @staticmethod
    def compact_arr(mat):
        return mat.reshape(BaseBackend.compact_shape(mat.shape))

    @staticmethod
    def compact_shape(shape):
        if len(shape) == 2:
            return shape[0], shape[1]
        elif len(shape) == 3:
            return shape[0], shape[1]*shape[2]
        else:
            return shape[0]*shape[1], shape[2]*shape[3]
