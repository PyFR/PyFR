# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
from pyfr.backends.openmp.compiler import GccSourceModule
import pyfr.backends.openmp.generator as generator
from pyfr.util import memoize


class OpenMPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        mod = GccSourceModule(src, self.backend.cfg)
        return mod.function(name, None, argtypes)


class OpenMPPointwiseKernelProvider(OpenMPKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = generator.OpenMPKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        class PointwiseKernel(ComputeKernel):
            def run(self, queue):
                fun(*arglst)

        return PointwiseKernel()
