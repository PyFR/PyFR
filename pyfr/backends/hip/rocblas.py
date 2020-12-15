# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.base import ComputeKernel
from pyfr.ctypesutil import load_library


# Possible RocBLAS exception types
RocBLASError = type('RocBLASError', (Exception,), {})
RocBLASInvalidHandle = type('RocBLASInvalidHandle', (RocBLASError,), {})
RocBLASInvalidPointer = type('RocBLASInvalidPointer', (RocBLASError,), {})
RocBLASInvalidSize = type('RocBLASInvalidSize', (RocBLASError,), {})
RocBLASInternalError = type('RocBLASInternalError', (RocBLASError,), {})
RocBLASInvalidValue = type('RocBLASInvalidValue', (RocBLASError,), {})


class RocBLASWrappers(object):
    # Possible return codes
    _statuses = {
        1: RocBLASInvalidHandle,
        3: RocBLASInvalidPointer,
        4: RocBLASInvalidSize,
        6: RocBLASInternalError,
        11: RocBLASInvalidValue
    }

    # Constants
    ROCBLAS_OPERATION_NONE = 111
    ROCBLAS_OPERATION_TRANSPOSE = 112

    def __init__(self):
        lib = load_library('rocblas')

        # rocblas_create_handle
        self.rocblas_create_handle = lib.rocblas_create_handle
        self.rocblas_create_handle.argtypes = [POINTER(c_void_p)]
        self.rocblas_create_handle.errcheck = self._errcheck

        # rocblas_destroy_handle
        self.rocblas_destroy_handle = lib.rocblas_destroy_handle
        self.rocblas_destroy_handle.argtypes = [c_void_p]
        self.rocblas_destroy_handle.errcheck = self._errcheck

        # rocblas_set_stream
        self.rocblas_set_stream = lib.rocblas_set_stream
        self.rocblas_set_stream.argtypes = [c_void_p, c_void_p]
        self.rocblas_set_stream.errcheck = self._errcheck

        # rocblas_dgemm
        self.rocblas_dgemm = lib.rocblas_dgemm
        self.rocblas_dgemm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int,
            POINTER(c_double), c_void_p, c_int, c_void_p, c_int,
            POINTER(c_double), c_void_p, c_int
        ]
        self.rocblas_dgemm.errcheck = self._errcheck

        # rocblas_sgemm
        self.rocblas_sgemm = lib.rocblas_sgemm
        self.rocblas_sgemm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int,
            POINTER(c_float), c_void_p, c_int, c_void_p, c_int,
            POINTER(c_float), c_void_p, c_int
        ]
        self.rocblas_sgemm.errcheck = self._errcheck

    def _errcheck(self, status, fn, args):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise RocBLASError


class HIPRocBLASKernels(object):
    def __init__(self, backend):
        # Load and wrap rocBLAS
        self._wrappers = RocBLASWrappers()

        # Init
        self._handle = c_void_p()
        self._wrappers.rocblas_create_handle(self._handle)

    def __del__(self):
        try:
            if self._handle:
                self._wrappers.rocblas_destroy_handle(self._handle)
        except AttributeError:
            pass

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        w = self._wrappers

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # RocBLAS expects inputs to be column-major (or Fortran order in
        # NumPy parlance).  However as C = A*B => C^T = (A*B)^T
        # = (B^T)*(A^T) with a little trickery we can multiply our
        # row-major matrices directly.
        m, n, k = b.ncol, a.nrow, a.ncol
        A, B, C = b, a, out

        # Do not transpose either A or B
        opA = opB = w.ROCBLAS_OPERATION_NONE

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.dtype == np.float64:
            rocblas_gemm = w.rocblas_dgemm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            rocblas_gemm = w.rocblas_sgemm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        class MulKernel(ComputeKernel):
            def run(iself, queue):
                w.rocblas_set_stream(self._handle, queue.hip_stream_comp)
                rocblas_gemm(self._handle, opA, opB, m, n, k,
                             alpha_ct, A, A.leaddim, B, B.leaddim,
                             beta_ct, C, C.leaddim)

        return MulKernel()
