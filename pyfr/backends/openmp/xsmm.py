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
            funcs = [ptr for blkptr in self._kerns.values() for ptr in blkptr]
            for f in funcs:
                self._wrappers.libxsmm_fsspmdm_destroy(f)

            self._wrappers.libxsmm_finalize()

    def mul(self, *mats, out, alpha=1.0, beta=0.0):
        *a_facs, b = mats

        # Ensure factors of A are compatible
        for al, ar in zip(a_facs[:-1], a_facs[1:]):
            if not al.nrow == ar.ncol:
                raise ValueError('Factors of a matrix are not compatible')

        # Ensure the matrices are compatible
        if a_facs[0].nrow != out.nrow or a_facs[-1].ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if any('const' not in a.tags for a in a_facs):
            raise NotSuitableError('libxsmm requires a constant a matrix')

        # Check n is suitable
        if b.leaddim % self._nmod != 0:
            raise NotSuitableError(f'libxsmm requires n % {self._nmod} = 0')

        # Check that beta is zero or one
        if beta != 0.0 and beta != 1.0:
            raise NotSuitableError('libxssm requires β = 0 or β = 1')

        # Dimensions
        ldb, ldc = b.leaddim, out.leaddim

        nfac = len(a_facs)

        blkptr = []
        for i, fac in enumerate(a_facs[::-1]):
            last = i == nfac - 1

            if last:
                _alpha, _beta = alpha, beta
                c_is_nt = (beta == 0 and
                           out.nbytes >= 32*1024**2 and
                           self.backend.alignb >= 64)
            else:
                _alpha, _beta, c_is_nt = 1.0, 0.0, 0

            ckey = (fac.mid, _alpha, _beta, c_is_nt, b.nblocks, ldb, ldc)

            # Check the JIT kernel cache
            try:
                blkptr.append(self._kerns[ckey])
            except KeyError:
                if self.backend.fpdtype == np.float64:
                    xsmm_dtype = 0
                    _alpha, _beta = c_double(_alpha), c_double(_beta)
                else:
                    xsmm_dtype = 1
                    _alpha, _beta = c_float(_alpha), c_float(_beta)

                mat = np.ascontiguousarray(fac.get())
                m, k = mat.shape

                # JIT and register a kernel for the current factor and sum the
                # result over out and use non-temporal stores only at last step
                blkptr.append(ptr := self._wrappers.libxsmm_fsspmdm_create(
                    xsmm_dtype, m, b.leaddim, k, k, ldb, ldc, byref(_alpha),
                    byref(_beta), c_is_nt, mat.ctypes.data
                ))

                if not ptr:
                    raise NotSuitableError('libxssm unable to JIT a kernel')

                # Update the cache
                self._kerns[ckey] = ptr

        # Determine size for the buffer array
        blocksz = max(a.nrow for a in a_facs)*out.leaddim

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('batch-gemm').render(
            nfac=nfac, blocksz=blocksz
        )

        # Build
        batch_gemm = self._build_kernel('batch_gemm', src, 'PiPiPi'+'P'*nfac)
        batch_gemm.set_args(self._exec_ptr, b.nblocks, b, b.blocksz,
                            out, out.blocksz, *blkptr)

        return OpenMPKernel(mats=[b, out], misc=[self], kernel=batch_gemm)
