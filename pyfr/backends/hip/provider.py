# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, Kernel)
from pyfr.backends.hip.compiler import SourceModule
from pyfr.backends.hip.generator import HIPKernelGenerator
from pyfr.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (-(-nrow // block[0]), -(-ncol // block[1]), 1)


class HIPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        # Compile the source code and retrieve the kernel
        return SourceModule(self.backend, src).get_function(name, argtypes)


class HIPPointwiseKernelProvider(HIPKernelProvider,
                                 BasePointwiseKernelProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._block1d = (64, 1, 1)
        self._block2d = (64, 4, 1)

        # Pass these block sizes to the generator
        class KernelGenerator(HIPKernelGenerator):
            block1d = self._block1d
            block2d = self._block2d

        self.kernel_generator_cls = KernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argmv):
        narglst = list(arglst)
        block = self._block1d if len(dims) == 1 else self._block2d
        grid = get_grid_for_block(block, dims[-1])

        # Identify any runtime arguments
        rtargs = [(i, k) for i, k in enumerate(arglst) if isinstance(k, str)]

        class PointwiseKernel(Kernel):
            def run(self, queue, **kwargs):
                for i, k in rtargs:
                    narglst[i] = kwargs[k]

                fun.exec_async(grid, block, queue.stream, *narglst)

        return PointwiseKernel(*argmv)
