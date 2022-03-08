# -*- coding: utf-8 -*-

from contextlib import contextmanager
from ctypes import c_void_p
import functools as ft
import hashlib
import itertools as it
import os
import pickle
import shutil

from pyfr.ctypesutil import get_libc_function


def memoize(meth):
    @ft.wraps(meth)
    def newmeth(self, *args, **kwargs):
        try:
            cache = self._memoize_cache_
        except AttributeError:
            cache = self._memoize_cache_ = {}

        if kwargs:
            key = (meth, args, tuple(kwargs.items()))
        else:
            key = (meth, args)

        try:
            return cache[key]
        except KeyError:
            pass
        except TypeError:
            key = (meth, pickle.dumps((args, kwargs)))

            try:
                return cache[key]
            except KeyError:
                pass

        res = cache[key] = meth(self, *args, **kwargs)
        return res

    return newmeth

class silence:
    def __init__(self, stdout=os.devnull, stderr=os.devnull):
        self.outfiles = (stdout, stderr)
        self.combine = (stdout == stderr)

        # Acquire a handle to fflush from libc
        self.libc_fflush = get_libc_function('fflush')
        self.libc_fflush.argtypes = [c_void_p]

    def __enter__(self):
        import sys
        self.sys = sys

        # Flush
        sys.__stdout__.flush()
        sys.__stderr__.flush()

        # Save
        self.saved_streams = [sys.__stdout__, sys.__stderr__]
        self.fds = [s.fileno() for s in self.saved_streams]
        self.saved_fds = [os.dup(f) for f in self.fds]

        # Open the redirects
        if self.combine:
            self.new_streams = [open(self.outfiles[0], 'wb', 0)]*2
        else:
            self.new_streams = [open(f, 'wb', 0) for f in self.outfiles]

        self.new_fds = [s.fileno() for s in self.new_streams]

        # Replace
        os.dup2(self.new_fds[0], self.fds[0])
        os.dup2(self.new_fds[1], self.fds[1])

    def __exit__(self, *args):
        sys = self.sys

        # Flush
        self.libc_fflush(None)

        # Restore
        os.dup2(self.saved_fds[0], self.fds[0])
        os.dup2(self.saved_fds[1], self.fds[1])

        sys.stdout, sys.stderr = self.saved_streams

        # Clean up
        self.new_streams[0].close()
        self.new_streams[1].close()

        os.close(self.saved_fds[0])
        os.close(self.saved_fds[1])


@contextmanager
def setenv(**kwargs):
    _env = os.environ.copy()
    os.environ.update(kwargs)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(_env)


@contextmanager
def chdir(dirname):
    cdir = os.getcwd()

    try:
        if dirname:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(cdir)


def subclasses(cls, just_leaf=False):
    sc = cls.__subclasses__()
    ssc = [g for s in sc for g in subclasses(s, just_leaf)]

    return [s for s in sc if not just_leaf or not s.__subclasses__()] + ssc


def subclass_where(cls, **kwargs):
    for s in subclasses(cls):
        for k, v in kwargs.items():
            if not hasattr(s, k) or getattr(s, k) != v:
                break
        else:
            return s

    attrs = ', '.join(f'{k} = {v}' for k, v in kwargs.items())
    raise KeyError(f'No subclasses of {cls.__name__} with attrs == ({attrs})')


def ndrange(*args):
    return it.product(*map(range, args))


def digest(*args, hash='sha256'):
    return getattr(hashlib, hash)(pickle.dumps(args)).hexdigest()


def rm(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


def mv(src, dst):
    shutil.move(src, dst)


def match_paired_paren(delim, n=5):
    open, close = delim
    ocset = f'[^{close}{open}]'

    lft = rf'{ocset}*?(?:\{open}'
    mid = rf'{ocset}*?'
    rgt = rf'\{close}{ocset}*?)*?'

    return lft*n + mid + rgt*n
