# -*- coding: utf-8 -*-

import sys

from ctypes import CDLL, POINTER, byref, c_int, c_double, c_float, c_void_p
from ctypes.util import find_library

import numpy as np

from pyfr.backends.base import ComputeKernel, traits
from pyfr.ctypesutil import platform_libname


class CublasError(Exception):
    pass


class CublasNotInitialized(CublasError):
    pass


class CublasAllocFailed(CublasError):
    pass


class CublasInvalidValue(CublasError):
    pass


class CublasArchMismatch(CublasError):
    pass


class CublasMappingError(CublasError):
    pass


class CublasExecutionFailed(CublasError):
    pass


class CublasInternalError(CublasError):
    pass


# Matrix operation types
class CublasOp(object):
    NONE = 0
    TRANS = 1
    CONJ_TRANS = 2


# Opaque CUBLAS handle pointer
class CublasHandle(c_void_p):
    pass


class CublasWrappers(object):
    # Possible return codes
    _statuses = {0x1: CublasNotInitialized,
                 0x3: CublasAllocFailed,
                 0x7: CublasInvalidValue,
                 0x8: CublasArchMismatch,
                 0xb: CublasMappingError,
                 0xd: CublasExecutionFailed,
                 0xe: CublasInternalError}

    def __init__(self, libname=None):
        libname = libname or platform_libname('cublas')

        try:
            lib = CDLL(libname)
        except OSError:
            raise RuntimeError('Unable to load CUBLAS')

        # cublasCreate
        self.cublasCreate = lib.cublasCreate_v2
        self.cublasCreate.argtypes = [POINTER(CublasHandle)]
        self.cublasCreate.errcheck = self._errcheck

        # cublasDestroy
        self.cublasDestroy = lib.cublasDestroy_v2
        self.cublasDestroy.argtypes = [CublasHandle]
        self.cublasDestroy.errcheck = self._errcheck

        # cublasSetStream
        self.cublasSetStream = lib.cublasSetStream_v2
        self.cublasSetStream.argtypes = [CublasHandle, c_void_p]
        self.cublasSetStream.errcheck = self._errcheck

        # cublasDgemm
        self.cublasDgemm = lib.cublasDgemm_v2
        self.cublasDgemm.argtypes = [CublasHandle, c_int, c_int,
                                     c_int, c_int, c_int,
                                     POINTER(c_double), c_void_p, c_int,
                                     c_void_p, c_int,
                                     POINTER(c_double), c_void_p, c_int]
        self.cublasDgemm.errcheck = self._errcheck

        # cublasSgemm
        self.cublasSgemm = lib.cublasSgemm_v2
        self.cublasSgemm.argtypes = [CublasHandle, c_int, c_int,
                                     c_int, c_int, c_int,
                                     POINTER(c_float), c_void_p, c_int,
                                     c_void_p, c_int,
                                     POINTER(c_float), c_void_p, c_int]
        self.cublasSgemm.errcheck = self._errcheck

        # cublasDnrm2
        self.cublasDnrm2 = lib.cublasDnrm2_v2
        self.cublasDnrm2.argtypes = [CublasHandle,
                                     c_int, c_void_p, c_int, POINTER(c_double)]
        self.cublasDnrm2.errcheck = self._errcheck

        # cublasSnrm2
        self.cublasSnrm2 = lib.cublasSnrm2_v2
        self.cublasSnrm2.argtypes = [CublasHandle,
                                     c_int, c_void_p, c_int, POINTER(c_float)]
        self.cublasSnrm2.errcheck = self._errcheck


    def _errcheck(self, status, fn, args):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise CublasError


class CudaCublasKernels(object):
    def __init__(self, backend, cfg):
        # Load and wrap cublas
        self._wrappers = CublasWrappers()

        # Init
        self._handle = CublasHandle()
        self._wrappers.cublasCreate(self._handle)

    def __del__(self):
        # PyCUDA registers an atexit handler to destroy the CUDA context
        # when Python terminates; however in exceptional circumstances this
        # can be *before* we are garbage collected (negating the need to call
        # cublasDestroy as we're terminating anyway).  We therefore need to
        # check for a valid context before calling cublasDestroy
        import pycuda.autoinit
        if pycuda.autoinit.context:
            self._wrappers.cublasDestroy(self._handle)

    @traits(a={'dense'})
    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # CUBLAS expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T
        # = (B^T)*(A^T) with a little trickery we can multiply our
        # row-major matrices directly.
        m, n, k = b.ncol, a.nrow, a.ncol
        A, B, C = b, a, out

        # α and β factors for C = α*(A*B) + β*C
        if a.dtype == np.float64:
            cublasgemm = self._wrappers.cublasDgemm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cublasgemm = self._wrappers.cublasSgemm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        class MulKernel(ComputeKernel):
            def run(iself, scomp, scopy):
                self._wrappers.cublasSetStream(self._handle, scomp.handle)
                cublasgemm(self._handle, CublasOp.NONE, CublasOp.NONE, m, n, k,
                           alpha_ct, A, A.leaddim, B, B.leaddim,
                           beta_ct, C, C.leaddim)

        return MulKernel()

    def nrm2(self, x):
        if x.dtype == np.float64:
            cublasnrm2 = self._wrappers.cublasDnrm2
            result = c_double()
        else:
            cublasnrm2 = self._wrappers.cublasSnrm2
            result = c_float()

        # Total number of elements (incl. slack)
        n = x.leaddim*x.nrow

        class Nrm2Kernel(ComputeKernel):
            @property
            def retval(iself):
                return result.value

            def run(iself, scomp, scopy):
                self._wrappers.cublasSetStream(self._handle, scomp.handle)
                cublasnrm2(self._handle, n, x, 1, result)

        return Nrm2Kernel()
