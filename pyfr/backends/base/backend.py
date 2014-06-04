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
        return self.matrix_cls(self, ioshape, initval, extent, tags)

    def matrix_rslice(self, mat, p, q):
        return self.matrix_rslice_cls(self, mat, p, q)

    def matrix_bank(self, mats, initbank=0, tags=set()):
        return self.matrix_bank_cls(self, mats, initbank, tags)

    def mpi_matrix(self, ioshape, initval=None, extent=None, tags=set()):
        return self.mpi_matrix_cls(self, ioshape, initval, extent, tags)

    def mpi_matrix_for_view(self, view, tags=set()):
        return self.mpi_matrix((view.nvrow, view.nvcol, view.n), tags=tags)

    def const_matrix(self, initval, extent=None, tags=set()):
        return self.const_matrix_cls(self, initval, extent, tags)

    def view(self, matmap, rcmap, stridemap=None, vshape=tuple(), tags=set()):
        return self.view_cls(self, matmap, rcmap, stridemap, vshape, tags)

    def mpi_view(self, matmap, rcmap, stridemap=None, vshape=tuple(),
                 tags=set()):
        return self.mpi_view_cls(self, matmap, rcmap, stridemap, vshape, tags)

    def kernel(self, name, *args, **kwargs):
        for prov in self._providers:
            kern = getattr(prov, name, None)
            if kern and issuitable(kern, *args, **kwargs):
                try:
                    return kern(*args, **kwargs)
                except NotImplementedError:
                    pass
        else:
            raise KeyError("'{}' has no providers".format(name))

    def queue(self):
        return self.queue_cls(self)

    def runall(self, sequence):
        self.queue_cls.runall(sequence)
