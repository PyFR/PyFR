# -*- coding: utf-8 -*-

from gimmik import generate_mm
import numpy as np

from pyfr.backends.base import ComputeKernel, NotSuitableError
from pyfr.backends.hip.provider import HIPKernelProvider, get_grid_for_block


class HIPGiMMiKKernels(HIPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Fetch the matrix and tally up the number of non-zeros
        arr = a.get()
        nnz, nuq = np.count_nonzero(arr), len(np.unique(np.abs(arr)))

        # Check that A is suitable
        if nuq > 28 and nnz / arr.size > 0.15:
            raise NotSuitableError('Matrix inappropriate GiMMiK')

        # Generate
        src = generate_mm(a.get(), dtype=a.dtype, platform='cuda',
                          alpha=alpha, beta=beta)
        src = src.replace('blockDim.x*blockIdx.x + threadIdx.x',
                          'hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x')

        # Build
        fun = self._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, b.ncol)

        class MulKernel(ComputeKernel):
            def run(self, queue):
                fun.exec_async(grid, block, queue.hip_stream_comp,
                               b.ncol, b, b.leaddim, out, out.leaddim)

        return MulKernel()
