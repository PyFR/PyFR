# -*- coding: utf-8 -*-

from ctypes import cast, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.base import ComputeKernel, NotSuitableError
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

        self.nblock = backend.cfg.getint('backend-openmp', 'libxsmm-block-sz',
                                         48)
        self.max_sz = backend.cfg.getint('backend-openmp', 'libxsmm-max-sz',
                                         125**2)

        # Ensure the block size is suitable
        if backend.fpdtype == np.float64 and self.nblock % 8 != 0:
            raise ValueError('libxsmm-block-sz must be a multiple of 8')
        elif backend.fpdtype == np.float32 and self.nblock % 16 != 0:
            raise ValueError('libxsmm-block-sz must be a multiple of 16')

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
            for blkptr, cleanptr in self._kerns.values():
                self._destroyfn(blkptr)

                if cleanptr:
                    self._destroyfn(cleanptr)

            self._wrappers.libxsmm_finalize()

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        nblock = self.nblock

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('libxsmm requires a constant a matrix')

        # Check n is suitable
        if b.ncol % self._nmod != 0:
            raise NotSuitableError(f'libxsmm requires n % {self._nmod} = 0')

        # Check that beta is zero or one
        if beta != 0.0 and beta != 1.0:
            raise NotSuitableError('libxssm requires β = 0 or β = 1')

        # Check the matrix is of a reasonable size
        if a.ncol*a.nrow > self.max_sz:
            raise NotSuitableError('Matrix too large for libxsmm')

        # Dimensions
        m, n, k = a.nrow, b.ncol, a.ncol
        lda, ldb, ldc = a.leaddim, b.leaddim, out.leaddim

        # Cache key
        ckey = (a.mid, alpha, beta, n, ldb, ldc)

        # Check the JIT kernel cache
        try:
            blkptr, cleanptr = self._kerns[ckey]
        except KeyError:
            c_is_nt = beta == 0 and self.backend.alignb >= 64

            # JIT and register an nblock size kernel for this matrix
            blkptr = self._createfn(m, nblock, k, lda, ldb, ldc, alpha,
                                    beta, c_is_nt, a)
            if not blkptr:
                raise NotSuitableError('libxssm unable to JIT a kernel')

            # If necessary, also JIT and register a clean-up kernel
            if n % nblock != 0:
                cleanptr = self._createfn(m, n % nblock, k, lda, ldb, ldc,
                                          alpha, beta, c_is_nt, a)
                if not cleanptr:
                    self._destroyfn(blkptr)
                    raise NotSuitableError('libxssm unable to JIT a kernel')
            else:
                cleanptr = 0

            # Update the cache
            self._kerns[ckey] = (blkptr, cleanptr)

        # Obtain a pointer to the execute function
        execptr = cast(self._execfn, c_void_p).value

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('par-xsmm').render()

        # Argument types for par_xsmm
        argt = [np.intp]*3 + [np.int32]*2 + [np.intp]*2

        # Build
        par_xsmm = self._build_kernel('par_xsmm', src, argt)

        class MulKernel(ComputeKernel):
            def run(iself, queue):
                par_xsmm(execptr, blkptr, cleanptr, n, nblock, b, out)

        return MulKernel()
