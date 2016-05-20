# -*- coding: utf-8 -*-

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
        gimmik_mm = generate_mm(a.get(), dtype=a.dtype, platform='c',
                                alpha=alpha, beta=beta)

        # Generate and build the OpenMP-wrapped GiMMiK kernel
        tpl = self.backend.lookup.get_template('par-gimmik')
        src = tpl.render(gimmik_mm=gimmik_mm)
        par_gimmik_mm = self._build_kernel('par_gimmik_mm', src,
                                           [np.int32] + [np.intp, np.int32]*2)

        class MulKernel(ComputeKernel):
            def run(self, queue):
                par_gimmik_mm(b.ncol, b, b.leaddim, out, out.leaddim)

        return MulKernel()
