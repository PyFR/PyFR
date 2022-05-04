# -*- coding: utf-8 -*-

from ctypes import Structure, byref, c_void_p

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, Kernel)
from pyfr.backends.openmp.generator import OpenMPKernelGenerator
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import memoize


class OpenMPKernelFunction:
    def __init__(self, fun, argcls):
        self.fun = fun
        self.kargs = argcls()

    def set_arg(self, i, v):
        setattr(self.kargs, f'arg{i}', getattr(v, '_as_parameter_', v))

    def set_args(self, *args, start=0):
        for i, arg in enumerate(args, start=start):
            self.set_arg(i, arg)

    def __call__(self):
        self.fun(byref(self.kargs))


class OpenMPKernelProvider(BaseKernelProvider):
    @memoize
    def _get_arg_cls(self, argtypes):
        fields = [(f'arg{i}', npdtype_to_ctypestype(arg))
                  for i, arg in enumerate(argtypes)]

        return type('ArgStruct', (Structure,), {'_fields_': fields})

    @memoize
    def _build_library(self, src):
        return self.backend.compiler.build(src)

    def _build_function(self, name, src, argtypes, restype=None):
        lib = self._build_library(src)
        return lib.function(name, restype,
                            [npdtype_to_ctypestype(arg) for arg in argtypes])

    def _build_kernel(self, name, src, argtypes):
        lib = self._build_library(src)
        fun = lib.function(name, None, [c_void_p])

        return OpenMPKernelFunction(fun, self._get_arg_cls(tuple(argtypes)))


class OpenMPPointwiseKernelProvider(OpenMPKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = OpenMPKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argmv):
        rtargs = []

        # Process the arguments
        for i, k in enumerate(arglst):
            if isinstance(k, str):
                rtargs.append((i, k))
            else:
                fun.set_arg(i, k)

        class PointwiseKernel(Kernel):
            def run(self, queue, **kwargs):
                for i, k in rtargs:
                    fun.set_arg(i, kwargs[k])

                fun()

        return PointwiseKernel(*argmv)
