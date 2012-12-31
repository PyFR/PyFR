# -*- coding: utf-8 -*-

import functools
import itertools
import contextlib

import os
import io
import cPickle as pickle

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

        key = (self.func, pickle.dumps(args[1:], 1), pickle.dumps(kw, 1))
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

    def __setattr__(self, attr, val):
        for x in self:
            setattr(x, attr, val)

    def __call__(self, *args, **kwargs):
        return proxylist([x(*args, **kwargs) for x in self])


@contextlib.contextmanager
def setenv(**kwargs):
    _env = os.environ.copy()
    os.environ.update(kwargs)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(_env)

def lazyprop(fn):
    attr = '_lazy_' + fn.__name__

    @property
    def newfn(self):
        try:
            return getattr(self, attr)
        except AttributeError:
            setattr(self, attr, fn(self))
            return getattr(self, attr)
    return newfn

def purge_lazyprops(obj):
    for attr in obj.__dict__:
        if attr.startswith('_lazy_'):
            del obj.__dict__[attr]

def all_subclasses(cls):
    return cls.__subclasses__()\
         + [g for s in cls.__subclasses__() for g in all_subclasses(s)]

def subclass_map(cls, attr):
    subcls = all_subclasses(cls)
    return {getattr(s, attr): s for s in subcls if hasattr(s, attr)}

def get_comm_rank_root():
    comm = MPI.COMM_WORLD
    return comm, comm.rank, 0

def get_local_rank():
    envs = ['OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK']

    for ev in envs:
        if ev in os.environ:
            return int(os.environ[ev])
    else:
        raise RuntimeError('Unknown/unsupported MPI implementation')

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


_range_eval_syms = {'__builtins__': None,
                    'range': lambda s,e,n: list(np.linspace(s, e, n))}
def range_eval(expr):
    return [float(t) for t in eval(expr, _range_eval_syms, None)]

_ctype_map = {np.float32: 'float', np.float64: 'double'}
def npdtype_to_ctype(dtype):
    return _ctype_map[np.dtype(dtype).type]

def ndrange(*args):
    return itertools.product(*map(xrange, args))
