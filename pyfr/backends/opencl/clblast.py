# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_float, c_size_t, c_void_p

import numpy as np

from pyfr.backends.base import ComputeKernel
from pyfr.ctypesutil import load_library


class CLBlastWrappers(object):
    def __init__(self):
        lib = load_library('clblast')

        # Constants
        self.CLBlastLayoutRowMajor = 101
        self.CLBlastLayoutColMajor = 102
        self.CLBlastTransposeNo = 111
        self.CLBlastTransposeYes = 112

        # CLBlastSgemm
        self.CLBlastSgemm = lib.CLBlastSgemm
        self.CLBlastSgemm.argtypes = [
            c_int, c_int, c_int,
            c_size_t, c_size_t, c_size_t, c_float,
            c_void_p, c_size_t, c_size_t,
            c_void_p, c_size_t, c_size_t, c_float,
            c_void_p, c_size_t, c_size_t,
            POINTER(c_void_p), c_void_p
        ]
        self.CLBlastSgemm.errcheck = self._errcheck

        # CLBlastDgemm
        self.CLBlastDgemm = lib.CLBlastDgemm
        self.CLBlastDgemm.argtypes = [
            c_int, c_int, c_int,
            c_size_t, c_size_t, c_size_t, c_double,
            c_void_p, c_size_t, c_size_t,
            c_void_p, c_size_t, c_size_t, c_double,
            c_void_p, c_size_t, c_size_t,
            POINTER(c_void_p), c_void_p
        ]
        self.CLBlastDgemm.errcheck = self._errcheck

    def _errcheck(self, status, fn, args):
        if status != 0:
            raise RuntimeError('CLBlast: {0}'.format(status))


class OpenCLCLBlastKernels(object):
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

        class MulKernel(ComputeKernel):
            def run(self, queue):
                qptr = c_void_p(queue.cl_queue_comp.int_ptr)
                clblastgemm(w.CLBlastLayoutRowMajor, w.CLBlastTransposeNo,
                            w.CLBlastTransposeNo, m, n, k, alpha,
                            a, 0, a.leaddim, b, 0, b.leaddim, beta,
                            out, 0, out.leaddim, qptr, None)

        return MulKernel()
