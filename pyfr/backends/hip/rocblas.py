from ctypes import (POINTER, byref, c_int, c_double, c_float, c_uint32,
                    c_void_p)

import numpy as np

from pyfr.backends.hip.provider import HIPKernel, HIPKernelProvider
from pyfr.ctypesutil import LibWrapper


# Possible RocBLAS exception types
class RocBLASError(Exception): pass
class RocBLASInvalidHandle(RocBLASError): pass
class RocBLASNotImplemented(RocBLASError): pass
class RocBLASInvalidPointer(RocBLASError): pass
class RocBLASInvalidSize(RocBLASError): pass
class RocBLASInternalError(RocBLASError): pass
class RocBLASInvalidValue(RocBLASError): pass


class RocBLASWrappers(LibWrapper):
    _libname = 'rocblas'

    # Error codes
    _statuses = {
        1: RocBLASInvalidHandle,
        2: RocBLASNotImplemented,
        3: RocBLASInvalidPointer,
        4: RocBLASInvalidSize,
        6: RocBLASInternalError,
        11: RocBLASInvalidValue
    }

    # Constants
    OPERATION_NONE = 111
    OPERATION_TRANSPOSE = 112
    DATATYPE_F32_R = 151
    DATATYPE_F64_R = 152
    GEMM_ALGO_SOLUTION_INDEX = 1

    # Functions
    _functions = [
        (c_int, 'rocblas_create_handle', POINTER(c_void_p)),
        (c_int, 'rocblas_destroy_handle', c_void_p),
        (c_int, 'rocblas_set_stream', c_void_p, c_void_p),
        (c_int, 'rocblas_gemm_ex_get_solutions', c_void_p, c_int, c_int, c_int,
         c_int, c_int, c_void_p, c_void_p, c_int, c_int, c_void_p,
         c_int, c_int, c_void_p, c_void_p, c_int, c_int, c_void_p,
         c_int, c_int, c_int, c_int, c_uint32, POINTER(c_int), POINTER(c_int)),
        (c_int, 'rocblas_gemm_ex', c_void_p, c_int, c_int, c_int,
         c_int, c_int, c_void_p, c_void_p, c_int, c_int, c_void_p,
         c_int, c_int, c_void_p, c_void_p, c_int, c_int, c_void_p,
         c_int, c_int, c_int, c_int, c_int, c_uint32)
    ]


class HIPRocBLASKernels(HIPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        # Load and wrap rocBLAS
        self._wrappers = RocBLASWrappers()

        # Init
        self._handle = c_void_p()
        self._wrappers.rocblas_create_handle(self._handle)

        # GEMM cache
        self._mul_cache = {}

        # Maximum number of solution indices to try
        self.nkerns = backend.cfg.getint('backend-hip', 'rocblas-nkerns', 2048)

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

        # Cache key
        ckey = (A.dtype, alpha, beta, m, n, k, A.leaddim, B.leaddim, C.leaddim)

        # Size checks
        if any(sz > 2**31 - 1 for sz in ckey[3:]):
            raise ValueError('Matrices too large for rocBLAS')

        # Do not transpose either A or B
        opA = opB = w.OPERATION_NONE

        # α and β factors for C = α*(A*B) + β*C
        if a.dtype == np.float64:
            rtype = w.DATATYPE_F64_R
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            rtype = w.DATATYPE_F32_R
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        def gemm(stream):
            w.rocblas_set_stream(h, stream)
            w.rocblas_gemm_ex(
                h, opA, opB, m, n, k, byref(alpha_ct), A, rtype, A.leaddim, B,
                rtype, B.leaddim, byref(beta_ct), C, rtype, C.leaddim, C,
                rtype, C.leaddim, rtype, w.GEMM_ALGO_SOLUTION_INDEX, algo, 0
            )

        try:
            algo, dt = self._mul_cache[ckey]
        except KeyError:
            def get_solutions(sidx):
                size_ct = c_int(len(sidx) if sidx is not None else 0)
                w.rocblas_gemm_ex_get_solutions(
                    h, opA, opB, m, n, k, byref(alpha_ct), A, rtype, A.leaddim,
                    B, rtype, B.leaddim, byref(beta_ct), C, rtype, C.leaddim,
                    C, rtype, C.leaddim, rtype, w.GEMM_ALGO_SOLUTION_INDEX, 0,
                    sidx, byref(size_ct)
                )
                return size_ct.value

            # Get applicable gemm algorithm solution indices
            sidx = (c_int * min(get_solutions(None), self.nkerns))()
            get_solutions(sidx)

            # Save a copy of the contents of the output matrix
            out_np = getattr(out, 'parent', out).get()

            best_kern = None

            # Benchmark suggested algorithms
            for algo in sidx:
                dt = self._benchmark(gemm)
                if best_kern is None or dt < best_kern[-1]:
                    best_kern = algo, dt

            # Restore the output matrix
            getattr(out, 'parent', out).set(out_np)

            # Update the cache
            self._mul_cache[ckey] = algo, dt = best_kern

        class MulKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                pass

            def run(self, stream):
                gemm(stream)

        return MulKernel(mats=[a, b, out], dt=dt)
