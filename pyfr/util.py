# -*- coding: utf-8 -*-

import functools
import itertools

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


class proxylist(list):
    def __init__(self, iterable):
        super(proxylist, self).__init__(iterable)

    def __getattr__(self, attr):
        return proxylist([getattr(x, attr) for x in self])

    def __call__(self, *args, **kwargs):
        return proxylist([x(*args, **kwargs) for x in self])


def all_subclasses(cls):
    return cls.__subclasses__()\
         + [g for s in cls.__subclasses__() for g in all_subclasses(s)]


_npeval_syms = {'__builtins__': None,
                'exp': np.exp, 'log': np.log,
                'sin': np.sin, 'asin': np.arcsin,
                'cos': np.cos, 'acos': np.arccos,
                'tan': np.tan, 'atan': np.arctan, 'atan2': np.arctan2,
                'abs': np.abs, 'pow': np.power, 'sqrt': np.sqrt,
                'pi': np.pi}
def npeval(expr, locals):
    # Allow '^' to be used for exponentiation
    expr = expr.replace('^', '**')

    return eval(expr, _npeval_syms, locals)

_ctype_map = {np.float32: 'float', np.float64: 'double'}
def npdtype_to_ctype(dtype):
    return _ctype_map[np.dtype(dtype).type]

def ndrange(*args):
    return itertools.product(*map(xrange, args))
