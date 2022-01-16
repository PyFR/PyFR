# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, Kernel)
from pyfr.backends.cuda.generator import CUDAKernelGenerator
from pyfr.backends.cuda.compiler import SourceModule
from pyfr.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (-(-nrow // block[0]), -(-ncol // block[1]), 1)


class CUDAKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        return SourceModule(self.backend, src).get_function(name, argtypes)


class CUDAPointwiseKernelProvider(CUDAKernelProvider,
                                  BasePointwiseKernelProvider):
    kernel_generator_cls = CUDAKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argmv):
        rtargs = []

        # Determine the block size
        if len(dims) == 1:
            block = (64, 1, 1)
        else:
            block = (64, 4, 1)

        # Use this to compute the grid size
        grid = get_grid_for_block(block, dims[-1])

        params = fun.make_params(grid, block)

        # Process the arguments
        for i, k in enumerate(arglst):
            if isinstance(k, str):
                rtargs.append((i, k))
            else:
                params.set_arg(i, k)

        class PointwiseKernel(Kernel):
            def run(self, queue, **kwargs):
                for i, k in rtargs:
                    params.set_arg(i, kwargs[k])

                fun.exec_async(queue.stream, params)

        return PointwiseKernel(*argmv)
