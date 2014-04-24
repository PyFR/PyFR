# -*- coding: utf-8 -*-

from contextlib import contextmanager
import functools as ft
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
    def __init__(self, *args):
        super(proxylist, self).__init__(*args)

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
    k, v = next(kwargs.iteritems())

    for s in subclasses(cls):
        if hasattr(s, k) and getattr(s, k) == v:
            return s


def ndrange(*args):
    return it.product(*map(xrange, args))


def rm(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    else:
        shutil.rmtree(path)
