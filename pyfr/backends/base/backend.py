# -*- coding: utf-8 -*-

from collections import defaultdict
from functools import cached_property, wraps
from itertools import count
import math
from weakref import WeakKeyDictionary, WeakValueDictionary

import numpy as np

from pyfr.backends.base.kernels import NotSuitableError
from pyfr.template import DottedTemplateLookup


def recordmat(fn):
    @wraps(fn)
    def newfn(self, *args, **kwargs):
        m = fn(self, *args, **kwargs)
        m.mid = next(self._mat_counter)
        self.mats[m.mid] = m

        return m
    return newfn


class BaseBackend:
    name = None

    def __init__(self, cfg):
        self.cfg = cfg

        # Numeric data type
        prec = cfg.get('backend', 'precision', 'double')
        if prec not in {'single', 'double'}:
            raise ValueError('Backend precision must be either single or '
                             'double')

        # Convert to a NumPy data type
        self.fpdtype = np.dtype(prec).type

        # Allocated matrices
        self.mats = WeakValueDictionary()
        self._mat_counter = count()

        # Aliases and extents
        self._pend_aliases = {}
        self._pend_extents = defaultdict(list)
        self._comm_extents = set()

        # Mapping from backend objects to memory extents
        self._obj_extents = WeakKeyDictionary()

    @cached_property
    def lookup(self):
        pkg = f'pyfr.backends.{self.name}.kernels'
        dfltargs = dict(fpdtype=self.fpdtype, soasz=self.soasz,
                        csubsz=self.csubsz, math=math)

        return DottedTemplateLookup(pkg, dfltargs)

    def malloc(self, obj, extent):
        # If no extent has been specified then autocommit
        if extent is None:
            # Perform the allocation
            data = self._malloc_impl(obj.nbytes)

            # Fire the callback
            obj.onalloc(data, 0)

            # Retain a (weak) reference to the allocated extent
            self._obj_extents[obj] = data
        # Otherwise defer the allocation
        else:
            # Check that the extent has not already been committed
            if extent in self._comm_extents:
                raise ValueError(f'Extent "{extent}" has already been '
                                 'allocated')

            # Append
            self._pend_extents[extent].append(obj)

            # Permit obj to be aliased
            self._pend_aliases[obj] = []

    def alias(self, obj, aobj):
        if obj.nbytes > aobj.nbytes:
            raise ValueError('Object too large to alias')

        try:
            obj.onalloc(self._obj_extents[aobj], aobj.offset)
        except KeyError:
            self._pend_aliases[aobj].append(obj)

    def commit(self):
        for reqs in self._pend_extents.values():
            # Determine the required allocation size
            sz = sum(obj.nbytes - (obj.nbytes % -self.alignb) for obj in reqs)

            # Perform the allocation
            data = self._malloc_impl(sz)

            offset = 0
            for obj in reqs:
                for aobj in [obj] + self._pend_aliases[obj]:
                    # Fire the objects allocation callback
                    aobj.onalloc(data, offset)

                    # Retain a (weak) reference to the allocated extent
                    self._obj_extents[aobj] = data

                # Increment the offset
                offset += obj.nbytes - (obj.nbytes % -self.alignb)

        # Mark the extents as committed and clear
        self._comm_extents.update(self._pend_extents)
        self._pend_aliases.clear()
        self._pend_extents.clear()

    def _malloc_impl(self, nbytes):
        pass

    @recordmat
    def const_matrix(self, initval, extent=None, tags=set()):
        return self.const_matrix_cls(self, initval, extent, tags)

    @recordmat
    def matrix(self, ioshape, initval=None, extent=None, aliases=None,
               tags=set()):
        return self.matrix_cls(self, ioshape, initval, extent, aliases, tags)

    @recordmat
    def matrix_slice(self, mat, ra, rb, ca, cb):
        return self.matrix_slice_cls(self, mat, ra, rb, ca, cb)

    @recordmat
    def xchg_matrix(self, ioshape, initval=None, extent=None, aliases=None,
                    tags=set()):
        return self.xchg_matrix_cls(self, ioshape, initval, extent, aliases,
                                    tags)

    def xchg_matrix_for_view(self, view, tags=set()):
        return self.xchg_matrix((view.nvrow, view.nvcol*view.n), tags=tags)

    def view(self, matmap, rmap, cmap, rstridemap=None, vshape=(), tags=set()):
        return self.view_cls(self, matmap, rmap, cmap, rstridemap, vshape,
                             tags)

    def xchg_view(self, matmap, rmap, cmap, rstridemap=None, vshape=(),
                  tags=set()):
        return self.xchg_view_cls(self, matmap, rmap, cmap, rstridemap,
                                  vshape, tags)

    def kernel(self, name, *args, **kwargs):
        for prov in self._providers:
            kern = getattr(prov, name, None)
            if kern:
                try:
                    return kern(*args, **kwargs)
                except NotSuitableError:
                    pass
        else:
            raise KeyError(f'Kernel "{name}" has no providers')

    def queue(self):
        return self.queue_cls(self)
