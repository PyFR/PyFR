# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
from pyfr.backends.cuda.generator import CUDAKernelGenerator
from pyfr.backends.cuda.compiler import SourceModule
from pyfr.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (int((nrow + (-nrow % block[0])) // block[0]),
            int((ncol + (-ncol % block[1])) // block[1]), 1)


class CUDAKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes, **kwargs):
        # Compile the source code and retrieve the function
        mod = SourceModule(self.backend, src)
        return mod.get_function(name, argtypes, **kwargs)


class CUDAPointwiseKernelProvider(CUDAKernelProvider,
                                  BasePointwiseKernelProvider):
    kernel_generator_cls = CUDAKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        cfg = self.backend.cfg

        # Determine the block size
        if len(dims) == 1:
            block = (cfg.getint('backend-cuda', 'block-1d', '64'), 1, 1)
        else:
            block = (cfg.getint('backend-cuda', 'block-2d', '128'), 1, 1)

        # Use this to compute the grid size
        grid = get_grid_for_block(block, dims[-1])

        class PointwiseKernel(ComputeKernel):
            if any(isinstance(arg, str) for arg in arglst):
                def run(self, queue, **kwargs):
                    fun.exec_async(grid, block, queue.stream_comp,
                                   *[kwargs.get(ka, ka) for ka in arglst])
            else:
                def run(self, queue, **kwargs):
                    fun.exec_async(grid, block, queue.stream_comp, *arglst)

        return PointwiseKernel()
