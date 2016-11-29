# -*- coding: utf-8 -*-

from ctypes import CDLL, cast, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.base import ComputeKernel
from pyfr.backends.openmp.provider import OpenMPKernelProvider


# Matrix orderings
class CBlasOrder(object):
    ROW_MAJOR = 101
    COL_MAJOR = 102


class CBlasTranspose(object):
    NO_TRANS = 111
    TRANS = 112
    CONJ_TRANS = 113


class CBlasWrappers(object):
    def __init__(self, libname):
        try:
            lib = CDLL(libname)
        except OSError:
            raise RuntimeError('Unable to load cblas')

        # cblas_dgemm
        self.cblas_dgemm = lib.cblas_dgemm
        self.cblas_dgemm.restype = None
        self.cblas_dgemm.argtypes = [
            c_int, c_int, c_int, c_int, c_int, c_int,
            c_double, c_void_p, c_int, c_void_p, c_int,
            c_double, c_void_p, c_int
        ]

        # cblas_sgemm
        self.cblas_sgemm = lib.cblas_sgemm
        self.cblas_sgemm.restype = None
        self.cblas_sgemm.argtypes = [
            c_int, c_int, c_int, c_int, c_int, c_int,
            c_float, c_void_p, c_int, c_void_p, c_int,
            c_float, c_void_p, c_int
        ]


class OpenMPCBLASKernels(OpenMPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        libname = backend.cfg.getpath('backend-openmp', 'cblas')
        libtype = backend.cfg.get('backend-openmp', 'cblas-type', 'parallel')

        if libtype not in {'serial', 'parallel'}:
            raise ValueError('cblas type must be serial or parallel')

        # Load and wrap cblas
        self._wrappers = CBlasWrappers(libname)
        self._cblas_type = libtype

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        m, n, k = a.nrow, b.ncol, a.ncol

        if a.dtype == np.float64:
            cblas_gemm = self._wrappers.cblas_dgemm
        else:
            cblas_gemm = self._wrappers.cblas_sgemm

        # If our BLAS library is single threaded then invoke our own
        # parallelization kernel which uses OpenMP to partition the
        # operation along b.ncol (which works extremely well for the
        # extremely long matrices encountered by PyFR).  Otherwise, we
        # let the BLAS library handle parallelization itself (which
        # may, or may not, use OpenMP).
        if self._cblas_type == 'serial':
            # Render the kernel template
            src = self.backend.lookup.get_template('par-gemm').render()

            # Argument types for par_gemm
            argt = [
                np.intp, np.int32, np.int32, np.int32,
                a.dtype, np.intp, np.int32, np.intp, np.int32,
                a.dtype, np.intp, np.int32
            ]

            # Build
            par_gemm = self._build_kernel('par_gemm', src, argt)

            # Pointer to the BLAS library GEMM function
            cblas_gemm_ptr = cast(cblas_gemm, c_void_p).value

            class MulKernel(ComputeKernel):
                def run(self, queue):
                    par_gemm(cblas_gemm_ptr, m, n, k, alpha, a, a.leaddim,
                             b, b.leaddim, beta, out, out.leaddim)
        else:
            class MulKernel(ComputeKernel):
                def run(self, queue):
                    cblas_gemm(CBlasOrder.ROW_MAJOR, CBlasTranspose.NO_TRANS,
                               CBlasTranspose.NO_TRANS, m, n, k,
                               alpha, a, a.leaddim, b, b.leaddim,
                               beta, out, out.leaddim)

        return MulKernel()
