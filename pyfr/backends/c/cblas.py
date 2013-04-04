 # -*- coding: utf-8 -*-

import os

from ctypes import CDLL, POINTER, byref, c_int, c_double, c_float, c_void_p
from ctypes.util import find_library

import numpy as np

from pyfr.backends.base import ComputeKernel, traits
from pyfr.ctypesutil import platform_libname


# Matrix orderings
class CBlasOrder(object):
    ROW_MAJOR = 101
    COL_MAJOR = 102


class CBlasTranspose(object):
    NO_TRANS = 111
    TRANS = 112
    CONJ_TRANS = 113


class CBlasWrappers(object):
    def __init__(self, libname):
        try:
            lib = CDLL(libname)
        except OSError:
            raise RuntimeError('Unable to load cblas')

        # cblas_dgemm
        self.cblas_dgemm = lib.cblas_dgemm
        self.cblas_dgemm.restype = None
        self.cblas_dgemm.argtypes = [c_int, c_int, c_int,
                                     c_int, c_int, c_int,
                                     c_double, c_void_p, c_int,
                                     c_void_p, c_int,
                                     c_double, c_void_p, c_int]

        # cblas_sgemm
        self.cblas_sgemm = lib.cblas_sgemm
        self.cblas_sgemm.restype = None
        self.cblas_sgemm.argtypes = [c_int, c_int, c_int,
                                     c_int, c_int, c_int,
                                     c_float, c_void_p, c_int,
                                     c_void_p, c_int,
                                     c_float, c_void_p, c_int]

        # cblas_dnrm2
        self.cblas_dnrm2 = lib.cblas_dnrm2
        self.cblas_dnrm2.restype = c_double
        self.cblas_dnrm2.argtypes = [c_int, c_void_p, c_int]

        # cblas_snrm2
        self.cblas_snrm2 = lib.cblas_snrm2
        self.cblas_snrm2.restype = c_float
        self.cblas_snrm2.argtypes = [c_int, c_void_p, c_int]



class CBlasKernels(object):
    def __init__(self, backend, cfg):
        libname = cfg.getpath('backend-c', 'cblas', abs=False)

        # Load and wrap cblas
        self._wrappers = CBlasWrappers(libname)

    @traits(a={'dense'})
    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        m, n, k = a.nrow, b.ncol, a.ncol

        if a.dtype == np.float64:
            cblas_gemm = self._wrappers.cblas_dgemm
        else:
            cblas_gemm = self._wrappers.cblas_sgemm

        class MulKernel(ComputeKernel):
            def run(self):
                cblas_gemm(CBlasOrder.ROW_MAJOR, CBlasTranspose.NO_TRANS,
                           CBlasTranspose.NO_TRANS, m, n, k,
                           alpha, a, a.leaddim, b, b.leaddim,
                           beta, out, out.leaddim)

        return MulKernel()

    def nrm2(self, x):
        if x.dtype == np.float64:
            cblas_nrm2 = self._wrappers.cblas_dnrm2
        else:
            cblas_nrm2 = self._wrappers.cblas_snrm2

        class Nrm2Kernel(ComputeKernel):
            @property
            def retval(self):
                return self._rv

            def run(self):
                self._rv = cblas_nrm2(x.leaddim*x.nrow, x, 1)

        return Nrm2Kernel()
