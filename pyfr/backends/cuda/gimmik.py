# -*- coding: utf-8 -*-

import gimmik.generator
import numpy as np

from pyfr.backends.base import ComputeKernel, traits
from pyfr.backends.cuda.provider import (CUDAKernelProvider,
                                         get_grid_for_block)


class CUDAGiMMiKKernels(CUDAKernelProvider):
    @traits(a={'dense'})
    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Generate
        mat = a.get()
        src = gimmik.generator.generateKernel(
            mat, 'cuda', alpha=alpha, beta=beta,
            double=a.dtype == np.float64, reduced=True,
        )

        # Build
        fun = self._build_kernel('gimmik_mm', src, 'PPiii')

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, b.ncol)

        class MulKernel(ComputeKernel):
            def run(self, queue):
                fun.prepared_async_call(grid, block, queue.cuda_stream_comp,
                                        b, out, b.ncol, b.leaddim,
                                        out.leaddim)

        return MulKernel()
