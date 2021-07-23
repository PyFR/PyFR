# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
from pyfr.backends.hip.compiler import SourceModule
from pyfr.backends.hip.generator import HIPKernelGenerator
from pyfr.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (int((nrow + (-nrow % block[0])) // block[0]),
            int((ncol + (-ncol % block[1])) // block[1]), 1)


class HIPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        # Compile the source code and retrieve the kernel
        return SourceModule(self.backend, src).get_function(name, argtypes)


class HIPPointwiseKernelProvider(HIPKernelProvider,
                                 BasePointwiseKernelProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Determine the block size for pointwise kernels
        cfg = self.backend.cfg
        self._blocksz = {
            1: (cfg.getint('backend-hip', 'block-1d', '64'), 1, 1),
            2: (cfg.getint('backend-hip', 'block-2d', '128'), 1, 1)
        }

        # Pass these to the HIP kernel generator
        class KernelGenerator(HIPKernelGenerator):
            block1d = self._blocksz[1]
            block2d = self._blocksz[2]

        self.kernel_generator_cls = KernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        block = self._blocksz[len(dims)]
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
