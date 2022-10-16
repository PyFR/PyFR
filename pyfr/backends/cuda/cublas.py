from ctypes import POINTER, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.cuda.provider import CUDAKernel
from pyfr.ctypesutil import LibWrapper


# Possible CUBLAS exception types
class CUBLASError(Exception): pass
class CUBLASNotInitialized(CUBLASError): pass
class CUBLASAllocFailed(CUBLASError): pass
class CUBLASInvalidValue(CUBLASError): pass
class CUBLASArchMismatch(CUBLASError): pass
class CUBLASMappingError(CUBLASError): pass
class CUBLASExecutionFailed(CUBLASError): pass
class CUBLASInternalError(CUBLASError): pass


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


class CUDACUBLASKernels:
    def __init__(self, backend):
        self.backend = backend
        self.handle = c_void_p()

        # Load and wrap CUBLAS
        self.lib = CUBLASWrappers()

        # Init
        self.lib.cublasCreate(self.handle)

    def __del__(self):
        if self.handle:
            self.lib.cublasDestroy(self.handle)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        cuda = self.backend.cuda
        w, h = self.lib, self.handle

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # CUBLAS expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T
        # = (B^T)*(A^T) with a little trickery we can multiply our
        # row-major matrices directly.
        m, n, k = b.ncol, a.nrow, a.ncol
        A, B, C = b, a, out

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.dtype == np.float64:
            cublasgemm = w.cublasDgemm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cublasgemm = w.cublasSgemm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        class MulKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                stream = cuda.create_stream()

                # Capture the execution of cuBLAS to obtain a graph
                stream.begin_capture()
                self.run(stream)
                gnode = stream.end_capture()

                # Embed this graph in our main graph
                return graph.graph.add_graph(gnode, deps)

            def run(self, stream):
                w.cublasSetStream(h, stream)
                cublasgemm(h, w.OP_N, w.OP_N, m, n, k,
                           alpha_ct, A, A.leaddim, B, B.leaddim,
                           beta_ct, C, C.leaddim)

        return MulKernel(mats=[a, b, out])
