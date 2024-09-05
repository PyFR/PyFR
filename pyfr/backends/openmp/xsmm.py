from ctypes import byref, cast, c_int, c_double, c_float, c_ulonglong, c_void_p
from weakref import finalize

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
         c_int, c_int, c_void_p, c_void_p, c_void_p, c_int, c_void_p),
        (None, 'libxsmm_fsspmdm_execute', c_void_p, c_void_p, c_void_p),
        (None, 'libxsmm_fsspmdm_destroy', c_void_p),
        (c_ulonglong, 'libxsmm_timer_tick')
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

    def _destroy_kern(self, k):
        blkptr, blkptr_nt = self._kerns.pop(k)

        self._wrappers.libxsmm_fsspmdm_destroy(blkptr)

        if blkptr_nt != blkptr:
            self._wrappers.libxsmm_fsspmdm_destroy(blkptr_nt)

    def __del__(self):
        if hasattr(self, '_wrappers'):
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
            raise NotSuitableError('libxsmm requires β = 0 or β = 1')

        # Index type
        ixdtype = self.backend.ixdtype

        # Dimensions
        ldb, ldc = b.leaddim, out.leaddim

        # Cache key
        ckey = (a.mid, alpha, beta, b.nblocks, ldb, ldc)

        # Check the JIT kernel cache
        try:
            blkptr, blkptr_nt = self._kerns[ckey]
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

            timer_tick = cast(self._wrappers.libxsmm_timer_tick, c_void_p)

            # Create a block leaddim size kernel for this matrix
            blkptr = self._wrappers.libxsmm_fsspmdm_create(
                xsmm_dtype, m, b.leaddim, k, k, ldb, ldc, byref(alpha),
                byref(beta), a_np.ctypes.data, False, timer_tick
            )
            if not blkptr:
                raise NotSuitableError('libxsmm unable to JIT a kernel')

            # Also consider creating a non-temporal kernel
            if c_is_nt:
                blkptr_nt = self._wrappers.libxsmm_fsspmdm_create(
                    xsmm_dtype, m, b.leaddim, k, k, ldb, ldc, byref(alpha),
                    byref(beta), a_np.ctypes.data, True, timer_tick
                )
                if not blkptr_nt:
                    raise NotSuitableError('libxsmm unable to JIT a kernel')
            else:
                blkptr_nt = blkptr

            # Update the cache
            self._kerns[ckey] = blkptr, blkptr_nt
            finalize(a, self._destroy_kern, ckey)

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('batch-gemm').render()

        # Build
        batch_gemm = self._build_kernel(
            'batch_gemm', src, [np.uintp]*3 + [np.uintp, ixdtype]*2,
            ['exec', 'blkptr', 'blkptr_nt', 'b', 'bsz', 'out', 'outsz']
        )
        batch_gemm.set_args(self._exec_ptr, blkptr, blkptr_nt, b, b.blocksz,
                            out, out.blocksz)
        batch_gemm.set_nblocks(b.nblocks)

        return OpenMPKernel(mats=[a, b, out], misc=[self], kernel=batch_gemm)
