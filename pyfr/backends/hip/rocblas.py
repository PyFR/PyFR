from ctypes import POINTER, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.hip.provider import HIPKernel, HIPKernelProvider
from pyfr.ctypesutil import LibWrapper


# Possible RocBLAS exception types
class RocBLASError(Exception): pass
class RocBLASInvalidHandle(RocBLASError): pass
class RocBLASInvalidPointer(RocBLASError): pass
class RocBLASInvalidSize(RocBLASError): pass
class RocBLASInternalError(RocBLASError): pass
class RocBLASInvalidValue(RocBLASError): pass


class RocBLASWrappers(LibWrapper):
    _libname = 'rocblas'

    # Error codes
    _statuses = {
        1: RocBLASInvalidHandle,
        3: RocBLASInvalidPointer,
        4: RocBLASInvalidSize,
        6: RocBLASInternalError,
        11: RocBLASInvalidValue
    }

    # Constants
    OPERATION_NONE = 111
    OPERATION_TRANSPOSE = 112

    # Functions
    _functions = [
        (c_int, 'rocblas_create_handle', POINTER(c_void_p)),
        (c_int, 'rocblas_destroy_handle', c_void_p),
        (c_int, 'rocblas_set_stream', c_void_p, c_void_p),
        (c_int, 'rocblas_dgemm', c_void_p, c_int, c_int, c_int, c_int, c_int,
         POINTER(c_double), c_void_p, c_int, c_void_p, c_int,
         POINTER(c_double), c_void_p, c_int),
        (c_int, 'rocblas_sgemm', c_void_p, c_int, c_int, c_int, c_int, c_int,
         POINTER(c_float), c_void_p, c_int, c_void_p, c_int,
         POINTER(c_float), c_void_p, c_int)
    ]


class HIPRocBLASKernels(HIPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        # Load and wrap rocBLAS
        self._wrappers = RocBLASWrappers()

        # Init
        self._handle = c_void_p()
        self._wrappers.rocblas_create_handle(self._handle)

        # Timing data cache
        self._mul_timing = {}

    def __del__(self):
        try:
            if self._handle:
                self._wrappers.rocblas_destroy_handle(self._handle)
        except AttributeError:
            pass

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        h, w = self._handle, self._wrappers

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
        opA = opB = w.OPERATION_NONE

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.dtype == np.float64:
            rocblas_gemm = w.rocblas_dgemm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            rocblas_gemm = w.rocblas_sgemm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        # Convenience wrapper
        def gemm(stream):
            w.rocblas_set_stream(h, stream)
            rocblas_gemm(h, opA, opB, m, n, k, alpha_ct, A, A.leaddim, B,
                         B.leaddim, beta_ct, C, C.leaddim)

        # Cache key
        ckey = (A.dtype, alpha, beta, m, n, k, A.leaddim, B.leaddim, C.leaddim)

        # Obtain the performance of the kernel
        try:
            dt = self._mul_timing[ckey]
        except KeyError:
            # Save a copy of the contents of the output matrix
            out_np = getattr(out, 'parent', out).get()

            # Benchmark the kernel and update the cache
            self._mul_timing[ckey] = dt = self._benchmark(gemm)

            # Restore the output matrix
            getattr(out, 'parent', out).set(out_np)

        class MulKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                pass

            def run(self, stream):
                gemm(stream)

        return MulKernel(mats=[a, b, out], dt=dt)
