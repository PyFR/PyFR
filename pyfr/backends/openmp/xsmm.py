# -*- coding: utf-8 -*-

from ctypes import cast, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.base import Kernel, NotSuitableError
from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.ctypesutil import LibWrapper


class XSMMWrappers(LibWrapper):
    _libname = 'xsmm'

    # Functions
    _functions = [
        (None, 'libxsmm_init'),
        (None, 'libxsmm_finalize'),
        (c_void_p, 'libxsmm_dfsspmdm_create', c_int, c_int, c_int, c_int,
         c_int, c_int, c_double, c_double, c_int, c_void_p),
        (c_void_p, 'libxsmm_sfsspmdm_create', c_int, c_int, c_int, c_int,
         c_int, c_int, c_float, c_float, c_int, c_void_p),
        (None, 'libxsmm_dfsspmdm_execute', c_void_p, c_void_p, c_void_p),
        (None, 'libxsmm_sfsspmdm_execute', c_void_p, c_void_p, c_void_p),
        (None, 'libxsmm_dfsspmdm_destroy', c_void_p),
        (None, 'libxsmm_sfsspmdm_destroy', c_void_p)
    ]


class OpenMPXSMMKernels(OpenMPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        # Kernel cache
        self._kerns = {}

        # Load and wrap libxsmm
        self._wrappers = XSMMWrappers()

        if backend.fpdtype == np.float64:
            self._nmod = 8
            self._createfn = self._wrappers.libxsmm_dfsspmdm_create
            self._execfn = self._wrappers.libxsmm_dfsspmdm_execute
            self._destroyfn = self._wrappers.libxsmm_dfsspmdm_destroy
        else:
            self._nmod = 16
            self._createfn = self._wrappers.libxsmm_sfsspmdm_create
            self._execfn = self._wrappers.libxsmm_sfsspmdm_execute
            self._destroyfn = self._wrappers.libxsmm_sfsspmdm_destroy

        # Init
        self._wrappers.libxsmm_init()

    def __del__(self):
        if hasattr(self, '_wrappers'):
            for blkptr in self._kerns.values():
                self._destroyfn(blkptr)

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

            # JIT and register an block leaddim size kernel for this matrix
            blkptr = self._createfn(m, b.leaddim, k, k, ldb, ldc, alpha,
                                    beta, c_is_nt, a_np.ctypes.data)
            if not blkptr:
                raise NotSuitableError('libxssm unable to JIT a kernel')

            # Update the cache
            self._kerns[ckey] = blkptr

        # Obtain a pointer to the execute function
        execptr = cast(self._execfn, c_void_p).value

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('batch-gemm').render()

        # Build
        batch_gemm = self._build_kernel('batch_gemm', src, 'PPiPiPi')
        batch_gemm.set_args(execptr, blkptr, b.nblocks, b, b.blocksz, out,
                            out.blocksz)

        class MulKernel(Kernel):
            def run(self, queue):
                batch_gemm()

        return MulKernel(mats=[b, out], misc=[self])
