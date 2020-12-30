# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.base import ComputeKernel
from pyfr.ctypesutil import LibWrapper


# Possible CUBLAS exception types
CUBLASError = type('CUBLASError', (Exception,), {})
CUBLASNotInitialized = type('CUBLASNotInitialized', (CUBLASError,), {})
CUBLASAllocFailed = type('CUBLASAllocFailed', (CUBLASError,), {})
CUBLASInvalidValue = type('CUBLASInvalidValue', (CUBLASError,), {})
CUBLASArchMismatch = type('CUBLASArchMismatch', (CUBLASError,), {})
CUBLASMappingError = type('CUBLASMappingError', (CUBLASError,), {})
CUBLASExecutionFailed = type('CUBLASExecutionFailed', (CUBLASError,), {})
CUBLASInternalError = type('CUBLASInternalError', (CUBLASError,), {})


class CUBLASWrappers(LibWrapper):
    _libname = 'cublas'

    # Error codes
    _statuses = {
        0x1: CUBLASNotInitialized,
        0x3: CUBLASAllocFailed,
        0x7: CUBLASInvalidValue,
        0x8: CUBLASArchMismatch,
        0xb: CUBLASMappingError,
        0xd: CUBLASExecutionFailed,
        0xe: CUBLASInternalError,
        '*': CUBLASError
    }

    # Constants
    OP_N = 0
    OP_T = 1
    OP_C = 2

    # Functions
    _functions = [
        (c_int, 'cublasCreate_v2', POINTER(c_void_p)),
        (c_int, 'cublasDestroy_v2', c_void_p),
        (c_int, 'cublasSetStream_v2', c_void_p, c_void_p),
        (c_int, 'cublasDgemm_v2', c_void_p, c_int, c_int, c_int, c_int, c_int,
         POINTER(c_double), c_void_p, c_int, c_void_p, c_int,
         POINTER(c_double), c_void_p, c_int),
        (c_int, 'cublasSgemm_v2', c_void_p, c_int, c_int, c_int, c_int, c_int,
         POINTER(c_float), c_void_p, c_int, c_void_p, c_int,
         POINTER(c_float), c_void_p, c_int)
    ]

    def _transname(self, name):
        return name[:-3]


class CUDACUBLASKernels(object):
    def __init__(self, backend):
        # Load and wrap CUBLAS
        self._wrappers = CUBLASWrappers()

        # Init
        self._handle = c_void_p()
        self._wrappers.cublasCreate(self._handle)

    def __del__(self):
        # PyCUDA registers an atexit handler to destroy the CUDA context
        # when Python terminates; however in exceptional circumstances this
        # can be *before* we are garbage collected (negating the need to call
        # cublasDestroy as we're terminating anyway).  We therefore need to
        # check for a valid context before calling cublasDestroy
        try:
            import pycuda.autoinit
            if pycuda.autoinit.context:
                self._wrappers.cublasDestroy(self._handle)
        except TypeError:
            pass

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        w = self._wrappers

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # CUBLAS expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T
        # = (B^T)*(A^T) with a little trickery we can multiply our
        # row-major matrices directly.
        m, n, k = b.ncol, a.nrow, a.ncol
        A, B, C = b, a, out

        # Do not transpose either A or B
        opA = opB = w.OP_N

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.dtype == np.float64:
            cublasgemm = w.cublasDgemm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cublasgemm = w.cublasSgemm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        class MulKernel(ComputeKernel):
            def run(iself, queue):
                w.cublasSetStream(self._handle, queue.cuda_stream_comp.handle)
                cublasgemm(self._handle, opA, opB, m, n, k,
                           alpha_ct, A, A.leaddim, B, B.leaddim,
                           beta_ct, C, C.leaddim)

        return MulKernel()
