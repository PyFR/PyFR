# -*- coding: utf-8 -*-

import functools
import cPickle

from mpi4py import MPI
import numpy as np

class memoize(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return functools.partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}

        key = (self.func, cPickle.dumps(args[1:], 1), cPickle.dumps(kw, 1))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)

        return res


_ctype_map = { np.float32: 'float', np.float64: 'double' }
def npdtype_to_ctype(dtype):
    return _ctype_map[np.dtype(dtype).type]

_mpitype_map = { np.float32: MPI.FLOAT, np.float64: MPI.DOUBLE }
def npdtype_to_mpitype(dtype):
    return _mpitype_map[np.dtype(dtype).type]
