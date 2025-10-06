from collections import defaultdict
from functools import cached_property, wraps
from itertools import count
import math
from weakref import WeakKeyDictionary, WeakValueDictionary

import numpy as np

from pyfr.backends.base.kernels import NotSuitableError
from pyfr.backends.base.makoutil import mfiltargs, mfiltid
from pyfr.template import DottedTemplateLookup


def recordmat(fn):
    @wraps(fn)
    def newfn(self, *args, **kwargs):
        m = fn(self, *args, **kwargs)

        if not hasattr(m, 'mid'):
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
        self.fpdtype_eps = np.finfo(self.fpdtype).eps
        self.fpdtype_max = np.finfo(self.fpdtype).max

        # Memory model
        match cfg.get('backend', 'memory-model', 'normal'):
            case 'normal':
                self.ixdtype = np.int32
            case 'large':
                self.ixdtype = np.int64
            case _:
                raise ValueError('Backend memory model must be either normal '
                                 'or large')

        # Autotuning improvement factor
        self.autotune_ifac = cfg.getfloat('backend', 'autotune-ifac', 0.95)

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
        dfltargs = {
            'fpdtype': self.fpdtype, 'ixdtype': self.ixdtype,
            'fpdtype_max': self.fpdtype_max, 'csubsz': self.csubsz,
            'soasz': self.soasz, 'math': math
        }

        lookup = DottedTemplateLookup(pkg, dfltargs)

        # Register source filters
        lookup.add_filter(mfiltargs)
        lookup.add_filter(mfiltid)

        return lookup

    def malloc(self, obj, extent):
        # If no extent has been specified then autocommit
        if extent is None:
            # Perform the allocation
            data = self._malloc_checked(obj.nbytes)

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
            data = self._malloc_checked(sz)

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

    def _malloc_checked(self, nbytes):
        if self.ixdtype == np.int32 and nbytes > 4*2**31 - 1:
            raise RuntimeError('Allocation too large for normal backend '
                               'memory-model')

        return self._malloc_impl(nbytes)

    def _malloc_impl(self, nbytes):
        pass

    @recordmat
    def const_matrix(self, initval, dtype=None, tags=set()):
        dtype = dtype or self.fpdtype

        # See if we have previously allocated an identical matrix
        for m in self.mats.values():
            if (isinstance(m, self.const_matrix_cls) and
                m.dtype == dtype and m.ioshape == initval.shape and
                tags.issubset(m.tags) and (m.get() == initval).all()):
                return m

        return self.const_matrix_cls(self, dtype, initval, tags)

    @recordmat
    def matrix(self, ioshape, initval=None, extent=None, aliases=None,
               tags=set(), dtype=None):
        dtype = dtype or self.fpdtype
        return self.matrix_cls(self, dtype, ioshape, initval, extent, aliases,
                               tags)

    @recordmat
    def matrix_slice(self, mat, ra, rb, ca, cb):
        return self.matrix_slice_cls(self, mat, ra, rb, ca, cb)

    @recordmat
    def xchg_matrix(self, ioshape, initval=None, extent=None, aliases=None,
                    tags=set()):
        return self.xchg_matrix_cls(self, self.fpdtype, ioshape, initval,
                                    extent, aliases, tags)

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
        best_kern = None

        # Loop through each kernel provider instance
        for prov in self._providers:
            # See if it can potentially provide the requested kernel
            kern_meth = getattr(prov, name, None)
            if kern_meth:
                ifac = self.autotune_ifac

                try:
                    # Ask the provider for the kernel
                    kern = kern_meth(*args, **kwargs)
                except NotSuitableError:
                    continue

                # Evaluate this kernel compared to the best seen so far
                if best_kern is None or kern.dt < ifac*best_kern.dt:
                    best_kern = kern

                    # If there is no benchmark data then short circut
                    if np.isnan(best_kern.dt):
                        return best_kern

        if best_kern is None:
            raise KeyError(f'Kernel "{name}" has no providers')

        return best_kern

    def ordered_meta_kernel(self, kerns):
        return self.ordered_meta_kernel_cls(kerns)

    def unordered_meta_kernel(self, kerns, splits=None):
        return self.unordered_meta_kernel_cls(kerns, splits)

    def graph(self):
        return self.graph_cls(self)
