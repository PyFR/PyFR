# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import ComputeKernel, NotSuitableError
from pyfr.backends.cuda.provider import (CUDAKernelProvider,
                                         get_grid_for_block)


class CUDAGiMMiKKernels(CUDAKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.max_nnz = backend.cfg.getint('backend-cuda', 'gimmik-max-nnz',
                                          512)

        try:
            from gimmik.generator import generateKernel

            self._gen_gimmik = generateKernel
            self.mul = self._mul_gimmik
        except ImportError:
            pass

    def _mul_gimmik(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Check that A is reasonably sparse
        if np.count_nonzero(a.get()) > self.max_nnz:
            raise NotSuitableError('Matrix too dense for GiMMiK')

        # Generate
        src = self._gen_gimmik(
            a.get(), 'cuda', alpha=alpha, beta=beta,
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
