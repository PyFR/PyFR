# -*- coding: utf-8 -*-

import sys

from ctypes import CDLL, POINTER, byref, c_int, c_double, c_float, c_void_p
from ctypes.util import find_library

import numpy as np

from pyfr.backends.cuda.provider import CudaKernelProvider
from pyfr.backends.cuda.queue import CudaComputeKernel

# Find the CUBLAS library
if sys.platform == 'linux2':
    _libcublasname = 'libcublas.so'
elif sys.platform == 'darwin':
    _libcublasname = 'libcublas.dylib'
elif sys.platform == 'Windows':
    _libcublasname = 'cublas.lib'
else:
    _libcublasname = find_library('cublas')
    if not _libcublasname:
        raise ImportError('Unsupport platform')

# Open up the library
try:
    _libcublas = CDLL(_libcublasname)
except OSError:
    raise ImportError('Library not found')


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


# Mapping between status codes and exceptions
_libcublas_status_map = {0x1: CublasNotInitialized,
                         0x3: CublasAllocFailed,
                         0x7: CublasInvalidValue,
                         0x8: CublasArchMismatch,
                         0xb: CublasMappingError,
                         0xd: CublasExecutionFailed,
                         0xe: CublasInternalError}


# Error handling
def _cublas_process_status(status, fn, args):
    if status != 0:
        try:
            raise _libcublas_status_map[status]
        except KeyError:
            raise CublasError


# Opaque CUBLAS handle pointer
class cublas_handle_t(c_void_p):
    pass


# Matrix operation types
class CublasOp(object):
    NONE = 0
    TRANS = 1
    CONJ_TRANS = 2


# Wrap the cublasCreate function
_cublasCreate = _libcublas.cublasCreate_v2
_cublasCreate.restype = c_int
_cublasCreate.argtypes = [POINTER(cublas_handle_t)]
_cublasCreate.errcheck = _cublas_process_status


# Wrap the cublasDestroy function
_cublasDestroy = _libcublas.cublasDestroy_v2
_cublasDestroy.restype = c_int
_cublasDestroy.argtypes = [cublas_handle_t]
_cublasDestroy.errcheck = _cublas_process_status


# Wrap the cublasSetStream function
_cublasSetStream = _libcublas.cublasSetStream_v2
_cublasSetStream.restype = c_int
_cublasSetStream.argtypes = [cublas_handle_t, c_void_p]
_cublasSetStream.errcheck = _cublas_process_status


# Wrap the cublasDgemm (double-precision general matrix multiply) function
_cublasDgemm = _libcublas.cublasDgemm_v2
_cublasDgemm.restype = c_int
_cublasDgemm.argtypes = [cublas_handle_t,
                         c_int, c_int,
                         c_int, c_int, c_int,
                         POINTER(c_double), c_void_p, c_int,
                         c_void_p, c_int,
                         POINTER(c_double), c_void_p, c_int]
_cublasDgemm.errcheck = _cublas_process_status


# Wrap the cublasSgemm (single-precision general matrix multiply) function
_cublasSgemm = _libcublas.cublasSgemm_v2
_cublasSgemm.restype = c_int
_cublasSgemm.argtypes = [cublas_handle_t,
                         c_int, c_int,
                         c_int, c_int, c_int,
                         POINTER(c_float), c_void_p, c_int,
                         c_void_p, c_int,
                         POINTER(c_float), c_void_p, c_int]
_cublasSgemm.errcheck = _cublas_process_status


class CudaCublasKernels(CudaKernelProvider):
    def __init__(self, backend):
        self._cublas = cublas_handle_t()
        _cublasCreate(self._cublas)

    def __del__(self):
        # PyCUDA registers an atexit handler to destroy the CUDA context
        # when Python terminates; however in exceptional circumstances this
        # can be *before* we are garbage collected (negating the need to call
        # cublasDestroy as we're terminating anyway).  We therefore need to
        # check for a valid context before calling cublasDestroy
        import pycuda.autoinit
        if pycuda.autoinit.context:
            _cublasDestroy(self._cublas)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # CUBLAS expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T
        # = (B^T)*(A^T) with a little trickery we can multiply our
        # row-major matrices directly.
        n, m, k = b.ncol, a.nrow, a.ncol
        A, B, C = b, a, out

        # α and β factors for C = α*(A*B) + β*C
        if a.dtype == np.float64:
            cublasgemm = _cublasDgemm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cublasgemm = _cublasSgemm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        class MulKernel(CudaComputeKernel):
            def run(iself, scomp, scopy):
                _cublasSetStream(self._cublas, scomp.handle)
                cublasgemm(self._cublas, CublasOp.NONE, CublasOp.NONE, n, m, k,
                           alpha_ct, A, A.leaddim, B, B.leaddim,
                           beta_ct, C, C.leaddim)

        return MulKernel()
