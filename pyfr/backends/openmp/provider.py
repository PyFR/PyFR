# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
from pyfr.backends.openmp.compiler import SourceModule
from pyfr.backends.openmp.generator import OpenMPKernelGenerator
from pyfr.util import memoize


class OpenMPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes, restype=None):
        mod = SourceModule(src, self.backend.cfg)
        return mod.function(name, restype, argtypes)


class OpenMPPointwiseKernelProvider(OpenMPKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = OpenMPKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        class PointwiseKernel(ComputeKernel):
            def run(self, queue, **kwargs):
                fun(*[kwargs.get(ka, ka) for ka in arglst])

        return PointwiseKernel()
