# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_float, c_size_t, c_void_p

import numpy as np

from pyfr.backends.opencl.provider import OpenCLKernel
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


class OpenCLCLBlastKernels:
    def __init__(self, backend):
        self.backend = backend
        self._wrappers = CLBlastWrappers()

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

        class MulKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                evt_ptr = c_void_p() if ret_evt else None

                # CLBlast does not support waiting for events so we
                # instead insert a barrier into the queue
                if wait_for:
                    queue.barrier(wait_for)

                qptr = c_void_p(int(queue))
                clblastgemm(w.LayoutRowMajor, w.TransposeNo, w.TransposeNo,
                            m, n, k, alpha, a, 0, a.leaddim, b, 0, b.leaddim,
                            beta, out, 0, out.leaddim, qptr, evt_ptr)

                if ret_evt:
                    return cl.event(evt_ptr)

        return MulKernel(mats=[a, b, out])
