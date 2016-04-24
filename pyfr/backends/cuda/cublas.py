# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.base import ComputeKernel
from pyfr.ctypesutil import load_library


# Possible CUBLAS exception types
CUBLASError = type('CUBLASError', (Exception,), {})
CUBLASNotInitialized = type('CUBLASNotInitialized', (CUBLASError,), {})
CUBLASAllocFailed = type('CUBLASAllocFailed', (CUBLASError,), {})
CUBLASInvalidValue = type('CUBLASInvalidValue', (CUBLASError,), {})
CUBLASArchMismatch = type('CUBLASArchMismatch', (CUBLASError,), {})
CUBLASMappingError = type('CUBLASMappingError', (CUBLASError,), {})
CUBLASExecutionFailed = type('CUBLASExecutionFailed', (CUBLASError,), {})
CUBLASInternalError = type('CUBLASInternalError', (CUBLASError,), {})


class CUBLASWrappers(object):
    # Possible return codes
    _statuses = {
        0x1: CUBLASNotInitialized,
        0x3: CUBLASAllocFailed,
        0x7: CUBLASInvalidValue,
        0x8: CUBLASArchMismatch,
        0xb: CUBLASMappingError,
        0xd: CUBLASExecutionFailed,
        0xe: CUBLASInternalError
    }

    def __init__(self):
        lib = load_library('cublas')

        # Constants
        self.CUBLAS_OP_N = 0
        self.CUBLAS_OP_T = 1
        self.CUBLAS_OP_C = 2

        # cublasCreate
        self.cublasCreate = lib.cublasCreate_v2
        self.cublasCreate.argtypes = [POINTER(c_void_p)]
        self.cublasCreate.errcheck = self._errcheck

        # cublasDestroy
        self.cublasDestroy = lib.cublasDestroy_v2
        self.cublasDestroy.argtypes = [c_void_p]
        self.cublasDestroy.errcheck = self._errcheck

        # cublasSetStream
        self.cublasSetStream = lib.cublasSetStream_v2
        self.cublasSetStream.argtypes = [c_void_p, c_void_p]
        self.cublasSetStream.errcheck = self._errcheck

        # cublasDgemm
        self.cublasDgemm = lib.cublasDgemm_v2
        self.cublasDgemm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int,
            POINTER(c_double), c_void_p, c_int, c_void_p, c_int,
            POINTER(c_double), c_void_p, c_int
        ]
        self.cublasDgemm.errcheck = self._errcheck

        # cublasSgemm
        self.cublasSgemm = lib.cublasSgemm_v2
        self.cublasSgemm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int,
            POINTER(c_float), c_void_p, c_int, c_void_p, c_int,
            POINTER(c_float), c_void_p, c_int
        ]
        self.cublasSgemm.errcheck = self._errcheck

    def _errcheck(self, status, fn, args):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise CUBLASError


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
        opA = opB = w.CUBLAS_OP_N

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
