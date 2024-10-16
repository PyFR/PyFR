from collections import namedtuple
from ctypes import (POINTER, Structure, byref, c_double, c_float, c_int,
                    c_int64, c_size_t, c_uint64, c_void_p, sizeof)

import numpy as np

from pyfr.backends.cuda.provider import CUDAKernel, CUDAKernelProvider
from pyfr.ctypesutil import LibWrapper


# Possible CUBLASLt exception types
class CUBLASLtError(Exception): pass
class CUBLASLtNotInitialized(CUBLASLtError): pass
class CUBLASLtAllocFailed(CUBLASLtError): pass
class CUBLASLtInvalidValue(CUBLASLtError): pass
class CUBLASLtArchMismatch(CUBLASLtError): pass
class CUBLASLtMappingError(CUBLASLtError): pass
class CUBLASLtExecutionFailed(CUBLASLtError): pass
class CUBLASLtInternalError(CUBLASLtError): pass
class CUBLASLtStatusNotSupported(CUBLASLtError): pass


class CUBLASLtMatmulAlgo(Structure):
    _fields_ = [('data', c_uint64 * 8)]


class CUBLASLtMatmulHeuristicResult(Structure):
    _fields_ = [
        ('algo', CUBLASLtMatmulAlgo),
        ('workspace_size', c_size_t),
        ('state', c_int),
        ('waves_count', c_float),
        ('reserved', c_int * 4),
    ]


class CUBLASLtWrappers(LibWrapper):
    _libname = 'cublasLt'

    # Error codes
    _statuses = {
        0x1: CUBLASLtNotInitialized,
        0x3: CUBLASLtAllocFailed,
        0x7: CUBLASLtInvalidValue,
        0x8: CUBLASLtArchMismatch,
        0xb: CUBLASLtMappingError,
        0xd: CUBLASLtExecutionFailed,
        0xe: CUBLASLtInternalError,
        0xf: CUBLASLtStatusNotSupported,
        '*': CUBLASLtError
    }

    # Constants
    COMPUTE_32F = 68
    COMPUTE_64F = 70
    MATMUL_DESC_TRANSA = (c_int, 3)
    MATMUL_DESC_TRANSB = (c_int, 4)
    MATMUL_DESC_TRANSC = (c_int, 5)
    MATMUL_PREF_MAX_WORKSPACE_BYTES = (c_size_t, 1)
    OP_N = 0
    OP_T = 1
    R_32F = 0
    R_64F = 1

    # Functions
    _functions = [
        (c_int, 'cublasLtCreate', POINTER(c_void_p)),
        (c_int, 'cublasLtDestroy', c_void_p),
        (c_int, 'cublasLtMatmul', c_void_p, c_void_p, c_void_p, c_void_p,
         c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
         c_void_p, POINTER(CUBLASLtMatmulAlgo), c_void_p, c_size_t, c_void_p),
        (c_int, 'cublasLtMatmulAlgoGetHeuristic', c_void_p, c_void_p, c_void_p,
         c_void_p, c_void_p, c_void_p, c_void_p, c_int,
         POINTER(CUBLASLtMatmulHeuristicResult), POINTER(c_int)),
        (c_int, 'cublasLtMatmulDescCreate', POINTER(c_void_p), c_int, c_int),
        (c_int, 'cublasLtMatmulDescDestroy', c_void_p),
        (c_int, 'cublasLtMatmulDescSetAttribute', c_void_p, c_int, c_void_p,
         c_size_t),
        (c_int, 'cublasLtMatrixLayoutCreate', POINTER(c_void_p), c_int,
         c_uint64, c_uint64, c_int64),
        (c_int, 'cublasLtMatrixLayoutDestroy', c_void_p),
        (c_int, 'cublasLtMatmulPreferenceCreate', POINTER(c_void_p)),
        (c_int, 'cublasLtMatmulPreferenceDestroy', c_void_p),
        (c_int, 'cublasLtMatmulPreferenceSetAttribute', c_void_p, c_int,
         c_void_p, c_size_t),
    ]


class _CUBLASLtBase:
    _fn_prefix = None

    def __init__(self, cublaslt, ptr, attrs=[]):
        self.cublaslt = cublaslt
        self._as_parameter_ = ptr.value

        for (ctype, k), v in attrs:
            setter = getattr(cublaslt.lib, f'{self._fn_prefix}SetAttribute')
            setter(self, k, byref(ctype(v)), sizeof(ctype))

    def __del__(self):
        try:
            getattr(self.cublaslt.lib, f'{self._fn_prefix}Destroy')(self)
        except AttributeError:
            pass


class CUBLASLtMatmulDesc(_CUBLASLtBase):
    _fn_prefix = 'cublasLtMatmulDesc'

    def __init__(self, cublaslt, ctype, dtype, attrs=[]):
        ptr = c_void_p()
        getattr(cublaslt.lib, f'{self._fn_prefix}Create')(ptr, ctype, dtype)

        super().__init__(cublaslt, ptr, attrs)


class CUBLASLtMatmulPreference(_CUBLASLtBase):
    _fn_prefix = 'cublasLtMatmulPreference'

    def __init__(self, cublaslt, attrs=[]):
        ptr = c_void_p()
        getattr(cublaslt.lib, f'{self._fn_prefix}Create')(ptr)

        super().__init__(cublaslt, ptr, attrs)


class CUBLASLtMatrixLayout(_CUBLASLtBase):
    _fn_prefix = 'cublasLtMatrixLayout'

    def __init__(self, cublaslt, mat, dtype):
        ptr = c_void_p()

        # PyFR is row-major and CUBLASLt is column-major
        getattr(cublaslt.lib, f'{self._fn_prefix}Create')(
            ptr, dtype, mat.ncol, mat.nrow, mat.leaddim
        )

        super().__init__(cublaslt, ptr)


GEMMDesc = namedtuple('GEMMDesc', ['mm', 'a', 'b', 'c', 'algo', 'ws_size'])


class CUDACUBLASLtKernels(CUDAKernelProvider):
    # Sets the default max workspace size based on recommendations
    WORKSPACE_MAX_SIZE = 32*1024**2

    def __init__(self, backend):
        super().__init__(backend)

        self.handle = c_void_p()

        # Load and wrap CUBLASLt
        self.lib = CUBLASLtWrappers()

        # Init
        self.lib.cublasLtCreate(self.handle)

        # Maximum number of algorithms to test
        self.nkerns = backend.cfg.getint('backend-cuda', 'cublas-nkerns', 512)

        # GEMM cache
        self._mul_cache = {}

    def __del__(self):
        if self.handle:
            self.lib.cublasLtDestroy(self.handle)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        cuda = self.backend.cuda
        w, h = self.lib, self.handle

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # CUBLASLt expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T
        # = (B^T)*(A^T) with a little trickery we can multiply our
        # row-major matrices directly.
        m, n, k = b.ncol, a.nrow, a.ncol
        A, B, C = b, a, out

        # Cache key
        ckey = (A.dtype, alpha, beta, m, n, k, A.leaddim, B.leaddim, C.leaddim)

        # α and β factors for D = α*(A*B) + β*C
        if a.dtype == np.float64:
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
            ctype = w.COMPUTE_64F
            dtype = w.R_64F
        else:
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)
            ctype = w.COMPUTE_32F
            dtype = w.R_32F

        # Convenience wrapper
        def gemm(stream):
            w.cublasLtMatmul(
                h, desc.mm, byref(alpha_ct), A, desc.a, B,
                desc.b, byref(beta_ct), C, desc.c, C, desc.c,
                byref(desc.algo), ws_ptr, desc.ws_size, stream
            )

        # Obtain the performance of the kernel
        try:
            desc, dt = self._mul_cache[ckey]
        except KeyError:
            # Create matrix layouts
            a_desc = CUBLASLtMatrixLayout(self, A, dtype)
            b_desc = CUBLASLtMatrixLayout(self, B, dtype)
            c_desc = CUBLASLtMatrixLayout(self, C, dtype)

            # Matmul descriptor
            mm_attrs = [(w.MATMUL_DESC_TRANSA, w.OP_N),
                        (w.MATMUL_DESC_TRANSB, w.OP_N),
                        (w.MATMUL_DESC_TRANSC, w.OP_N)]
            mm_desc = CUBLASLtMatmulDesc(self, ctype, dtype, attrs=mm_attrs)

            # Allocate some temporary workspace for benchmarks
            ws_ptr = cuda.mem_alloc(self.WORKSPACE_MAX_SIZE)

            # Heuristic preference descriptor
            pref_attrs = [(w.MATMUL_PREF_MAX_WORKSPACE_BYTES, ws_ptr.nbytes)]
            pref = CUBLASLtMatmulPreference(self, attrs=pref_attrs)

            # Get matmul heuristics
            heurs = (CUBLASLtMatmulHeuristicResult * self.nkerns)()
            nreturn = c_int()
            w.cublasLtMatmulAlgoGetHeuristic(
                h, mm_desc, a_desc, b_desc, c_desc, c_desc, pref,
                self.nkerns, heurs, byref(nreturn)
            )

            # Save a copy of the contents of the output matrix
            out_np = getattr(out, 'parent', out).get()

            # Benchmark the kernel
            best_kern = None
            for heur in heurs[:nreturn.value]:
                desc = GEMMDesc(
                    mm_desc, a_desc, b_desc, c_desc, heur.algo,
                    heur.workspace_size
                )

                try:
                    dt = self._benchmark(gemm)
                except CUBLASLtStatusNotSupported:
                    continue

                if best_kern is None or dt < best_kern[-1]:
                    best_kern = desc, dt

            # Restore the output matrix
            getattr(out, 'parent', out).set(out_np)

            # Update the cache
            self._mul_cache[ckey] = desc, dt = best_kern

        # Allocate workspace for gemm
        ws_ptr = cuda.mem_alloc(desc.ws_size) if desc.ws_size else None

        class MulKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                stream = cuda.create_stream()

                # Capture the execution of CUBLASLt to obtain a graph
                stream.begin_capture()
                gemm(stream)
                gnode = stream.end_capture()

                # Embed this graph in our main graph
                return graph.graph.add_graph(gnode, deps)

            def run(self, stream):
                gemm(stream)

        return MulKernel(mats=[a, b, out], dt=dt, misc=[ws_ptr])
