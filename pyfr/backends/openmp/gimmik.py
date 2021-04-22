# -*- coding: utf-8 -*-

from ctypes import cast, c_void_p

from gimmik import generate_mm
import numpy as np

from pyfr.backends.base import ComputeKernel, NotSuitableError
from pyfr.backends.openmp.provider import OpenMPKernelProvider


class OpenMPGiMMiKKernels(OpenMPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.max_nnz = backend.cfg.getint('backend-openmp', 'gimmik-max-nnz',
                                          512)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Check that A is reasonably sparse
        if np.count_nonzero(a.get()) > self.max_nnz:
            raise NotSuitableError('Matrix too dense for GiMMiK')

        # Generate the GiMMiK kernel
        src = generate_mm(a.get(), dtype=a.dtype, platform='c',
                          alpha=alpha, beta=beta)
        gimmik_mm = self._build_kernel('gimmik_mm', src,
                                       [np.int32] + [np.intp, np.int32]*2)
        gimmik_ptr = cast(gimmik_mm, c_void_p).value

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('batch-gemm').render(
            lib='gimmik'
        )

        # Argument types for batch_gemm
        argt = [np.intp] + [np.int32]*2 + [np.intp, np.int32]*2

        # Build
        batch_gemm = self._build_kernel('batch_gemm', src, argt)

        class MulKernel(ComputeKernel):
            def run(self, queue):
                batch_gemm(gimmik_ptr, b.leaddim, b.nblocks, b, b.blocksz, out,
                           out.blocksz)

        return MulKernel()
