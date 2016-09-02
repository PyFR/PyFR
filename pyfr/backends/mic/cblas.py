# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import ComputeKernel
from pyfr.backends.mic.provider import MICKernelProvider


class MICCBLASKernels(MICKernelProvider):
    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        m, n, k = a.nrow, b.ncol, a.ncol

        # Data type
        if a.dtype == np.float64:
            cblas_gemm = 'cblas_dgemm'
        else:
            cblas_gemm = 'cblas_sgemm'

        # Render the kernel template
        tpl = self.backend.lookup.get_template('gemm')
        src = tpl.render(cblas_gemm=cblas_gemm)

        # Argument types for gemm
        argt = [
            np.intp, np.int32, np.int32, np.int32,
            a.dtype, np.intp, np.int32, np.intp, np.int32,
            a.dtype, np.intp, np.int32
        ]

        # Build
        gemm = self._build_kernel('gemm', src, argt)

        class MulKernel(ComputeKernel):
            def run(self, queue):
                queue.mic_stream_comp.invoke(
                    gemm, m, n, k, a.data, b.data, out.data, a.leaddim,
                    b.leaddim, out.leaddim, alpha, beta
                )

        return MulKernel()
