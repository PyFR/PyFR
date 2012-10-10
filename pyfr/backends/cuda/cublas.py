# -*- coding: utf-8 -*-

import sys

from ctypes import CDLL, POINTER, byref, c_int, c_double, c_void_p
from ctypes.util import find_library

from pyfr.exc import PyFRError

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


class CublasError(PyFRError):
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
_libcublas_status_map = { 0x1: CublasNotInitialized,
                          0x3: CublasAllocFailed,
                          0x7: CublasInvalidValue,
                          0x8: CublasArchMismatch,
                          0xb: CublasMappingError,
                          0xd: CublasExecutionFailed,
                          0xe: CublasInternalError }

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
    NONE       = 0
    TRANS      = 1
    CONJ_TRANS = 2


# Wrap the cublasCreate function
_cublasCreate = _libcublas.cublasCreate_v2
_cublasCreate.restype  = c_int
_cublasCreate.argtypes = [POINTER(cublas_handle_t)]
_cublasCreate.errcheck = _cublas_process_status

# Wrap the cublasDestroy function
_cublasDestroy = _libcublas.cublasDestroy_v2
_cublasDestroy.restype  = c_int
_cublasDestroy.argtypes = [cublas_handle_t]
_cublasDestroy.errcheck = _cublas_process_status

# Wrap the cublasSetStream function
_cublasSetStream = _libcublas.cublasSetStream_v2
_cublasSetStream.restype  = c_int
_cublasSetStream.argtypes = [cublas_handle_t, c_void_p]
_cublasSetStream.errcheck = _cublas_process_status

# Wrap the cublasDgemm (double-precision general matrix multiply) function
_cublasDgemm = _libcublas.cublasDgemm_v2
_cublasDgemm.restype = c_int
_cublasDgemm.argtypes = [cublas_handle_t,
                         c_int, c_int,
                         c_int, c_int, c_int,
                         c_void_p, c_void_p, c_int,
                         c_void_p, c_int,
                         c_void_p, c_void_p, c_int]
_cublasDgemm.errcheck = _cublas_process_status

# Wrap the cublasDaxpy (y[i] += α*x[i]) function
_cublasDaxpy = _libcublas.cublasDaxpy_v2
_cublasDaxpy.restype  = c_int
_cublasDaxpy.argtypes = [cublas_handle_t, c_int, c_void_p,
                         c_void_p, c_int, c_void_p, c_int]
_cublasDaxpy.errcheck = _cublas_process_status


class CudaCublasKernels(CudaKernelProvider):
    # Useful constants when making CUBLAS calls
    _zero = c_double(0.0)
    _one  = c_double(1.0)

    def __init__(self, backend):
        self._cublas = cublas_handle_t()
        _cublasCreate(byref(self._cublas))

    def __del__(self):
        # PyCUDA registers an atexit handler to destroy the CUDA context
        # when Python terminates; however in exceptional circumstances this
        # can be *before* we are garbage collected (negating the need to call
        # cublasDestroy as we're terminating anyway).  We therefore need to
        # check for a valid context before calling cublasDestroy
        import pycuda.autoinit
        if pycuda.autoinit.context:
            _cublasDestroy(self._cublas)

    def mul(self, a, b, out):
        # Ensure the matrices are compatible
        if a.order != b.order or a.order != out.order or\
           a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # CUBLAS expects inputs to be column-major (or Fortran order in numpy
        # parlance).  However as C = A*B => C^T = (A*B)^T = (B^T)*(A^T) with
        # a little trickery we can multiply row-major matrices directly
        if a.order == 'F':
            n, m, k = a.nrow, b.ncol, a.ncol
            A, B, C = a, b, out
        else:
            n, m, k = b.ncol, a.nrow, a.ncol
            A, B, C = b, a, out

        class MulKernel(CudaComputeKernel):
            def __call__(iself, stream):
                _cublasSetStream(self._cublas, stream.handle)
                _cublasDgemm(self._cublas, CublasOp.NONE, CublasOp.NONE,
                             n, m, k, byref(self._one),
                             int(A.data), A.leaddim,
                             int(B.data), B.leaddim, byref(self._zero),
                             int(C.data), C.leaddim)

        return MulKernel()

    def ipadd(self, y, alpha, x):
        # Ensure x and y have not only the same logical dimensions but also
        # the same layout in memory; this is required on account of vector
        # addition being used to add the matrices
        if (y.majdim, y.mindim, y.leaddim) != (x.majdim, x.mindim, x.leaddim):
            raise TypeError('Incompatible matrix types for in-place addition')

        elecnt  = y.leaddim*y.majdim
        alpha_d = c_double(alpha)

        class IpaddKernel(CudaComputeKernel):
            def __call__(iself, stream):
                _cublasSetStream(self._cublas, stream.handle)
                _cublasDaxpy(self._cublas, elecnt, byref(alpha_d),
                             int(x.data), 1, int(y.data), 1)

        return IpaddKernel()
