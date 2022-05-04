# -*- coding: utf-8 -*-

from gimmik import generate_mm
import numpy as np

from pyfr.backends.base import Kernel, NotSuitableError
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

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, b.ncol)

        # Generate
        src = generate_mm(arr, a.dtype, 'hip', alpha=alpha, beta=beta,
                          n=b.ncol, ldb=b.leaddim, ldc=out.leaddim)

        # Build
        fun = self._build_kernel('gimmik_mm', src, [np.intp, np.intp])

        class MulKernel(Kernel):
            def run(self, queue):
                fun.exec_async(grid, block, queue.stream, b, out)

        return MulKernel()
