# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, Kernel)
from pyfr.backends.openmp.generator import OpenMPKernelGenerator
from pyfr.util import memoize


class OpenMPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes, restype=None):
        mod = self.backend.compiler.build(src)
        return mod.function(name, restype, argtypes)


class OpenMPPointwiseKernelProvider(OpenMPKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = OpenMPKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argmv):
        class PointwiseKernel(Kernel):
            if any(isinstance(arg, str) for arg in arglst):
                def run(self, queue, **kwargs):
                    fun(*[kwargs.get(ka, ka) for ka in arglst])
            else:
                def run(self, queue, **kwargs):
                    fun(*arglst)

        return PointwiseKernel(*argmv)
