# -*- coding: utf-8 -*-

from contextlib import contextmanager
import functools as ft
import io
import itertools as it
import os
import cPickle as pickle
import shutil


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

        key = (self.func, pickle.dumps(args[1:], 1), pickle.dumps(kwargs, 1))

        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kwargs)

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


def lazyprop(fn):
    attr = '_lazy_' + fn.__name__

    @property
    @ft.wraps(fn)
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
    return (cls.__subclasses__() +
            [g for s in cls.__subclasses__() for g in all_subclasses(s)])


def subclass_map(cls, attr):
    subcls = all_subclasses(cls)
    return {getattr(s, attr): s for s in subcls if hasattr(s, attr)}


def ndrange(*args):
    return it.product(*map(xrange, args))


def rm(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    else:
        shutil.rmtree(path)
