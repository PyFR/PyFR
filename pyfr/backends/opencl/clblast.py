from ctypes import POINTER, c_int, c_double, c_float, c_size_t, c_void_p

import numpy as np

from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider
from pyfr.ctypesutil import LibWrapper


# Possible CLBlast exception types
class CLBlastError(Exception): pass


class CLBlastWrappers(LibWrapper):
    _libname = 'clblast'

    # Error codes
    _statuses = {
        '*': CLBlastError
    }

    # Constants
    LayoutRowMajor = 101
    LayoutColMajor = 102
    TransposeNo = 111
    TransposeYes = 112

    # Functions
    _functions = [
        (c_int, 'CLBlastDgemm', c_int, c_int, c_int, c_size_t, c_size_t,
         c_size_t, c_double, c_void_p, c_size_t, c_size_t, c_void_p, c_size_t,
         c_size_t, c_double, c_void_p, c_size_t, c_size_t, POINTER(c_void_p),
         POINTER(c_void_p)),
        (c_int, 'CLBlastSgemm', c_int, c_int, c_int, c_size_t, c_size_t,
         c_size_t, c_float, c_void_p, c_size_t, c_size_t, c_void_p, c_size_t,
         c_size_t, c_float, c_void_p, c_size_t, c_size_t, POINTER(c_void_p),
         POINTER(c_void_p))
    ]


class OpenCLCLBlastKernels(OpenCLKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self._wrappers = CLBlastWrappers()

        # Timing data cache
        self._mul_timing = {}

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        w = self._wrappers
        cl = self.backend.cl

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        m, n, k = a.nrow, b.ncol, a.ncol

        if a.dtype == np.float64:
            clblastgemm = w.CLBlastDgemm
        else:
            clblastgemm = w.CLBlastSgemm

        # Cache key
        ckey = (a.dtype, alpha, beta, m, n, k, a.leaddim, b.leaddim,
                out.leaddim)

        # Obtain the performance of the kernel
        try:
            dt = self._mul_timing[ckey]
        except KeyError:
            # Save a copy of the contents of the output matrix
            out_np = getattr(out, 'parent', out).get()

            def gemm(queue):
                evt_ptr, q_ptr = c_void_p(), c_void_p(int(queue))

                clblastgemm(w.LayoutRowMajor, w.TransposeNo, w.TransposeNo,
                            m, n, k, alpha, a, 0, a.leaddim, b, 0, b.leaddim,
                            beta, out, 0, out.leaddim, q_ptr, evt_ptr)

                return cl.event(evt_ptr)

            # Benchmark the kernel and update the cache
            self._mul_timing[ckey] = dt = self._benchmark(gemm)

            # Restore the output matrix
            getattr(out, 'parent', out).set(out_np)

        class MulKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                evt_ptr = c_void_p() if ret_evt else None

                # CLBlast does not support waiting for events so we
                # instead insert a barrier into the queue
                if wait_for:
                    queue.barrier(wait_for)

                q_ptr = c_void_p(int(queue))
                clblastgemm(w.LayoutRowMajor, w.TransposeNo, w.TransposeNo,
                            m, n, k, alpha, a, 0, a.leaddim, b, 0, b.leaddim,
                            beta, out, 0, out.leaddim, q_ptr, evt_ptr)

                if ret_evt:
                    return cl.event(evt_ptr)

        return MulKernel(mats=[a, b, out], dt=dt)
