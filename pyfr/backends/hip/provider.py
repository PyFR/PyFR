# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
from pyfr.backends.hip.compiler import SourceModule
import pyfr.backends.hip.generator as generator
from pyfr.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (int((nrow + (-nrow % block[0])) // block[0]),
            int((ncol + (-ncol % block[1])) // block[1]), 1)


class HIPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        # Compile the source code and retrieve the kernel
        return SourceModule(self.backend, src).function(name, argtypes)


class HIPPointwiseKernelProvider(HIPKernelProvider,
                                 BasePointwiseKernelProvider):
    kernel_generator_cls = generator.HIPKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        cfg = self.backend.cfg

        # Determine the block size
        if len(dims) == 1:
            block = (cfg.getint('backend-hip', 'block-1d', '64'), 1, 1)
        else:
            block = cfg.getliteral('backend-hip', 'block-2d', '128, 1')
            block += (1,)

        # Use this to compute the grid size
        grid = get_grid_for_block(block, *dims[::-1])

        class PointwiseKernel(ComputeKernel):
            def run(self, queue, **kwargs):
                narglst = [kwargs.get(ka, ka) for ka in arglst]
                fun.exec_async(grid, block, queue.hip_stream_comp, *narglst)

        return PointwiseKernel()
