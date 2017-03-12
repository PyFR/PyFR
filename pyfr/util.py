# -*- coding: utf-8 -*-

from contextlib import contextmanager
from ctypes import CDLL, c_void_p
import functools as ft
import hashlib
import itertools as it
import os
import pickle
import shutil

from pyfr.ctypesutil import find_libc


class memoize(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func if instance is None else ft.partial(self, instance)

    def __call__(self, *args, **kwargs):
        instance = args[0]

        try:
            cache = instance._memoize_cache
        except AttributeError:
            cache = instance._memoize_cache = {}

        key = (self.func, pickle.dumps(args[1:]), pickle.dumps(kwargs))

        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kwargs)

        return res


class proxylist(list):
    def __getattr__(self, attr):
        return proxylist(getattr(x, attr) for x in self)

    def __setattr__(self, attr, val):
        for x in self:
            setattr(x, attr, val)

    def __call__(self, *args, **kwargs):
        return proxylist(x(*args, **kwargs) for x in self)


class silence(object):
    def __init__(self, stdout=os.devnull, stderr=os.devnull):
        self.outfiles = stdout, stderr
        self.combine = (stdout == stderr)

        # Acquire a handle to fflush from libc
        self.libc_fflush = CDLL(find_libc()).fflush
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


class lazyprop(object):
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, instance, owner):
        if instance is None:
            return None

        value = self.fn(instance)
        setattr(instance, self.fn.__name__, value)

        return value


def subclasses(cls, just_leaf=False):
    sc = cls.__subclasses__()
    ssc = [g for s in sc for g in subclasses(s, just_leaf)]

    return [s for s in sc if not just_leaf or not s.__subclasses__()] + ssc


def subclass_where(cls, **kwargs):
    k, v = next(iter(kwargs.items()))

    for s in subclasses(cls):
        if hasattr(s, k) and getattr(s, k) == v:
            return s

    raise KeyError("No subclasses of {0} with cls.{1} == '{2}'"
                   .format(cls.__name__, k, v))


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
