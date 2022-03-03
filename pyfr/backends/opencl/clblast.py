# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_float, c_size_t, c_void_p

import numpy as np

from pyfr.backends.base import Kernel
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
         c_void_p),
        (c_int, 'CLBlastSgemm', c_int, c_int, c_int, c_size_t, c_size_t,
         c_size_t, c_float, c_void_p, c_size_t, c_size_t, c_void_p, c_size_t,
         c_size_t, c_float, c_void_p, c_size_t, c_size_t, POINTER(c_void_p),
         c_void_p)
    ]


class OpenCLCLBlastKernels:
    def __init__(self, backend):
        # Load and wrap CLBlast
        self._wrappers = CLBlastWrappers()

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        w = self._wrappers

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        m, n, k = a.nrow, b.ncol, a.ncol

        if a.dtype == np.float64:
            clblastgemm = w.CLBlastDgemm
        else:
            clblastgemm = w.CLBlastSgemm

        class MulKernel(Kernel):
            def run(self, queue):
                qptr = c_void_p(int(queue.cmd_q))
                clblastgemm(w.LayoutRowMajor, w.TransposeNo, w.TransposeNo,
                            m, n, k, alpha, a, 0, a.leaddim, b, 0, b.leaddim,
                            beta, out, 0, out.leaddim, qptr, None)

        return MulKernel()
