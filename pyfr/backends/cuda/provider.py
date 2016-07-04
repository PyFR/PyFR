# -*- coding: utf-8 -*-

from pycuda import compiler, driver

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
import pyfr.backends.cuda.generator as generator
from pyfr.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (int((nrow + (-nrow % block[0])) // block[0]),
            int((ncol + (-ncol % block[1])) // block[1]))


class CUDAKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        # Compile the source code and retrieve the kernel
        fun = compiler.SourceModule(src).get_function(name)

        # Prepare the kernel for execution
        fun.prepare(argtypes)

        # Declare a preference for L1 cache over shared memory
        fun.set_cache_config(driver.func_cache.PREFER_L1)

        return fun


class CUDAPointwiseKernelProvider(CUDAKernelProvider,
                                  BasePointwiseKernelProvider):
    kernel_generator_cls = generator.CUDAKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        cfg = self.backend.cfg

        # Determine the block size
        if len(dims) == 1:
            block = (cfg.getint('backend-cuda', 'block-1d', '64'), 1, 1)
        else:
            block = cfg.getliteral('backend-cuda', 'block-2d', '128, 1')
            block += (1,)

        # Use this to compute the grid size
        grid = get_grid_for_block(block, *dims[::-1])

        class PointwiseKernel(ComputeKernel):
            def run(self, queue, **kwargs):
                narglst = [kwargs.get(ka, ka) for ka in arglst]
                fun.prepared_async_call(grid, block, queue.cuda_stream_comp,
                                        *narglst)

        return PointwiseKernel()
