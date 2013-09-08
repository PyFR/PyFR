# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

from pyfr.util import proxylist


class Kernel(object):
    """Bound kernel abstract base class"""
    __metaclass__ = ABCMeta

    def __call__(self, *args):
        return self, args

    @property
    def retval(self):
        return None

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class ComputeKernel(Kernel):
    pass


class MPIKernel(Kernel):
    pass


def iscomputekernel(kernel):
    return isinstance(kernel, ComputeKernel)


def ismpikernel(kernel):
    return isinstance(kernel, MPIKernel)


class _MetaKernel(object):
    def __init__(self, kernels):
        self._kernels = proxylist(kernels)

    def run(self, *args, **kwargs):
        self._kernels.run(*args, **kwargs)


class ComputeMetaKernel(_MetaKernel, ComputeKernel):
    pass


class MPIMetaKernel(_MetaKernel, MPIKernel):
    pass
