# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
from pyfr.backends.mic.compiler import MICSourceModule
import pyfr.backends.mic.generator as generator
from pyfr.util import memoize


class MICKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes, restype=None):
        mod = MICSourceModule(src, self.backend.dev, self.backend.cfg)
        return mod.function(name, argtypes, restype)


class MICPointwiseKernelProvider(MICKernelProvider,
                                 BasePointwiseKernelProvider):
    kernel_generator_cls = generator.MICKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        class PointwiseKernel(ComputeKernel):
            def run(self, queue, **kwargs):
                narglst = [kwargs.get(ka, ka) for ka in arglst]
                narglst = [getattr(arg, 'dev_ptr', arg) for arg in narglst]
                narglst = [getattr(arg, 'data', arg) for arg in narglst]
                queue.mic_stream_comp.invoke(fun, *narglst)

        return PointwiseKernel()
