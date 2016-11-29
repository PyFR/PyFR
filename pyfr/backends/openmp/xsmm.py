# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.base import ComputeKernel, NotSuitableError
from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.ctypesutil import load_library


class XSMMWrappers(object):
    def __init__(self):
        lib = load_library('xsmm')

        self.libxsmm_init = lib.libxsmm_init
        self.libxsmm_init.argtypes = []
        self.libxsmm_init.restype = None

        self.libxsmm_finalize = lib.libxsmm_finalize
        self.libxsmm_finalize.argtypes = []
        self.libxsmm_finalize.restype = None

        self.libxsmm_dmmdispatch = lib.libxsmm_dmmdispatch
        self.libxsmm_dmmdispatch.argtypes = [
            c_int, c_int, c_int,
            POINTER(c_int), POINTER(c_int), POINTER(c_int),
            POINTER(c_double), POINTER(c_double),
            POINTER(c_int), POINTER(c_int)
        ]
        self.libxsmm_dmmdispatch.restype = c_void_p

        self.libxsmm_smmdispatch = lib.libxsmm_smmdispatch
        self.libxsmm_smmdispatch.argtypes = [
            c_int, c_int, c_int,
            POINTER(c_int), POINTER(c_int), POINTER(c_int),
            POINTER(c_float), POINTER(c_float),
            POINTER(c_int), POINTER(c_int)
        ]
        self.libxsmm_smmdispatch.restype = c_void_p


class OpenMPXSMMKernels(OpenMPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.nblock = backend.soasz
        self.max_sz = backend.cfg.getint('backend-openmp', 'libxsmm-max-sz',
                                         125**2)

        # Load and wrap libxsmm
        self._wrappers = XSMMWrappers()

        # Init
        self._wrappers.libxsmm_init()

    def __del__(self):
        if hasattr(self, '_wrappers'):
            self._wrappers.libxsmm_finalize()

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        w = self._wrappers

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check n is divisible by the blocking factor
        if b.ncol % self.nblock != 0:
            raise NotSuitableError('libxsmm requires n % nblock = 0')

        # Check the matrix is of a reasonable size
        if a.ncol*a.nrow > self.max_sz:
            raise NotSuitableError('Matrix too large for libxsmm')

        # Dimensions
        m, n, k = a.nrow, b.ncol, a.ncol
        lda, ldb, ldc = c_int(a.leaddim), c_int(b.leaddim), c_int(out.leaddim)

        # α and β factors for C = α*(A*B) + β*C
        if a.dtype == np.float64:
            mmdispatch = w.libxsmm_dmmdispatch
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            mmdispatch = w.libxsmm_smmdispatch
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        # JIT a column major multiplication kernel for a single N-block
        xsmm_ptr = mmdispatch(self.nblock, m, k, ldb, lda, ldc,
                              alpha_ct, beta_ct, None, c_int(0))

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('par-xsmm').render()

        # Argument types for par_xsmm
        argt = [np.intp, np.int32, np.int32, np.intp, np.intp, np.intp]

        # Build
        par_xsmm = self._build_kernel('par_xsmm', src, argt)

        class MulKernel(ComputeKernel):
            def run(iself, queue):
                par_xsmm(xsmm_ptr, n, self.nblock, a, b, out)

        return MulKernel()
