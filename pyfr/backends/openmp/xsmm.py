# -*- coding: utf-8 -*-

from ctypes import byref, cast, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.base import NotSuitableError
from pyfr.backends.openmp.provider import OpenMPKernel, OpenMPKernelProvider
from pyfr.ctypesutil import LibWrapper


class XSMMWrappers(LibWrapper):
    _libname = 'xsmm'

    # Functions
    _functions = [
        (None, 'libxsmm_init'),
        (None, 'libxsmm_finalize'),
        (c_void_p, 'libxsmm_fsspmdm_create', c_int, c_int, c_int, c_int, c_int,
         c_int, c_int, c_void_p, c_void_p, c_int, c_void_p),
        (None, 'libxsmm_fsspmdm_execute', c_void_p, c_void_p, c_void_p),
        (None, 'libxsmm_fsspmdm_destroy', c_void_p)
    ]


class OpenMPXSMMKernels(OpenMPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        # Kernel cache
        self._kerns = {}

        # Load and wrap libxsmm
        self._wrappers = w = XSMMWrappers()

        self._exec_ptr = cast(w.libxsmm_fsspmdm_execute, c_void_p).value
        self._nmod = 8 if backend.fpdtype == np.float64 else 16

        # Init
        w.libxsmm_init()

    def __del__(self):
        if hasattr(self, '_wrappers'):
            for blkptr in self._kerns.values():
                self._wrappers.libxsmm_fsspmdm_destroy(blkptr)

            self._wrappers.libxsmm_finalize()

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('libxsmm requires a constant a matrix')

        # Check n is suitable
        if b.leaddim % self._nmod != 0:
            raise NotSuitableError(f'libxsmm requires n % {self._nmod} = 0')

        # Check that beta is zero or one
        if beta != 0.0 and beta != 1.0:
            raise NotSuitableError('libxssm requires β = 0 or β = 1')

        # Dimensions
        ldb, ldc = b.leaddim, out.leaddim

        # Cache key
        ckey = (a.mid, alpha, beta, b.nblocks, ldb, ldc)

        # Check the JIT kernel cache
        try:
            blkptr = self._kerns[ckey]
        except KeyError:
            c_is_nt = (beta == 0 and
                       out.nbytes >= 32*1024**2 and
                       self.backend.alignb >= 64)

            a_np = np.ascontiguousarray(a.get())
            m, k = a_np.shape

            if self.backend.fpdtype == np.float64:
                xsmm_dtype = 0
                alpha, beta = c_double(alpha), c_double(beta)
            else:
                xsmm_dtype = 1
                alpha, beta = c_float(alpha), c_float(beta)

            # JIT and register an block leaddim size kernel for this matrix
            blkptr = self._wrappers.libxsmm_fsspmdm_create(
                xsmm_dtype, m, b.leaddim, k, k, ldb, ldc, byref(alpha),
                byref(beta), c_is_nt, a_np.ctypes.data
            )
            if not blkptr:
                raise NotSuitableError('libxssm unable to JIT a kernel')

            # Update the cache
            self._kerns[ckey] = blkptr

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('batch-gemm').render()

        # Build
        batch_gemm = self._build_kernel('batch_gemm', src, 'PPiPiPi')
        batch_gemm.set_args(self._exec_ptr, blkptr, b.nblocks, b, b.blocksz,
                            out, out.blocksz)

        return OpenMPKernel(mats=[b, out], misc=[self], kernel=batch_gemm)
