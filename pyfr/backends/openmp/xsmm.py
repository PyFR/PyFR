# -*- coding: utf-8 -*-

from ctypes import cast, c_int, c_double, c_float, c_void_p

import numpy as np

from pyfr.backends.base import ComputeKernel, NotSuitableError
from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.ctypesutil import load_library


class XSMMWrappers(object):
    def __init__(self):
        lib = load_library('xsmm')

        # libxsmm_init
        self.libxsmm_init = lib.libxsmm_init
        self.libxsmm_init.argtypes = []
        self.libxsmm_init.restype = None

        # libxsmm_finalize
        self.libxsmm_finalize = lib.libxsmm_finalize
        self.libxsmm_finalize.argtypes = []
        self.libxsmm_finalize.restype = None

        # libxsmm_dfsspmdm_create
        self.libxsmm_dfsspmdm_create = lib.libxsmm_dfsspmdm_create
        self.libxsmm_dfsspmdm_create.argtypes = [
            c_int, c_int, c_int, c_int, c_int, c_int,
            c_double, c_double, c_void_p
        ]
        self.libxsmm_dfsspmdm_create.restype = c_void_p

        # libxsmm_sfsspmdm_create
        self.libxsmm_sfsspmdm_create = lib.libxsmm_sfsspmdm_create
        self.libxsmm_sfsspmdm_create.argtypes = [
            c_int, c_int, c_int, c_int, c_int, c_int,
            c_float, c_float, c_void_p
        ]
        self.libxsmm_sfsspmdm_create.restype = c_void_p

        # libxsmm_dfsspmdm_execute
        self.libxsmm_dfsspmdm_execute = lib.libxsmm_dfsspmdm_execute
        self.libxsmm_dfsspmdm_execute.argtypes = [
            c_void_p, c_void_p, c_void_p
        ]
        self.libxsmm_dfsspmdm_execute.restype = None

        # libxsmm_sfsspmdm_execute
        self.libxsmm_sfsspmdm_execute = lib.libxsmm_sfsspmdm_execute
        self.libxsmm_sfsspmdm_execute.argtypes = [
            c_void_p, c_void_p, c_void_p
        ]
        self.libxsmm_sfsspmdm_execute.restype = None

        # libxsmm_dfsspmdm_destroy
        self.libxsmm_dfsspmdm_destroy = lib.libxsmm_dfsspmdm_destroy
        self.libxsmm_dfsspmdm_destroy.argtypes = [c_void_p]
        self.libxsmm_dfsspmdm_destroy.restype = None

        # libxsmm_sfsspmdm_destroy
        self.libxsmm_sfsspmdm_destroy = lib.libxsmm_sfsspmdm_destroy
        self.libxsmm_sfsspmdm_destroy.argtypes = [c_void_p]
        self.libxsmm_sfsspmdm_destroy.restype = None


class OpenMPXSMMKernels(OpenMPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.nblock = backend.cfg.getint('backend-openmp', 'libxsmm-block-sz',
                                         48)
        self.max_sz = backend.cfg.getint('backend-openmp', 'libxsmm-max-sz',
                                         125**2)

        # Ensure the block size is divisible by 16
        if self.nblock % 16 != 0:
            raise ValueError('libxsmm-block-sz must be a multiple of 16')

        # Active kernel list
        self._kerns = []

        # Load and wrap libxsmm
        self._wrappers = XSMMWrappers()

        # Init
        self._wrappers.libxsmm_init()

    def __del__(self):
        if hasattr(self, '_wrappers'):
            for kern, destroy in self._kerns:
                destroy(kern)

            self._wrappers.libxsmm_finalize()

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        w = self._wrappers
        nblock = self.nblock

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('libxsmm requires a constant a matrix')

        # Check n is divisible by 16
        if b.ncol % 16 != 0:
            raise NotSuitableError('libxsmm requires n % 16 = 0')

        # Check that beta is zero or one
        if beta != 0.0 and beta != 1.0:
            raise NotSuitableError('libxssm requires β = 0 or β = 1')

        # Check the matrix is of a reasonable size
        if a.ncol*a.nrow > self.max_sz:
            raise NotSuitableError('Matrix too large for libxsmm')

        # Dimensions
        m, n, k = a.nrow, b.ncol, a.ncol
        lda, ldb, ldc = a.leaddim, b.leaddim, out.leaddim

        # Precision specific functions
        if a.dtype == np.float64:
            create = w.libxsmm_dfsspmdm_create
            execute = w.libxsmm_dfsspmdm_execute
            destroy = w.libxsmm_dfsspmdm_destroy

            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            create = w.libxsmm_sfsspmdm_create
            execute = w.libxsmm_sfsspmdm_execute
            destroy = w.libxsmm_sfsspmdm_destroy

            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        # Get the A matrix
        a_np = a.get()

        # JIT and register an nblock size kernel for this A matrix
        blockk_ptr = create(m, nblock, k, lda, ldb, ldc, alpha_ct,
                            beta_ct, a_np.ctypes.data)
        self._kerns.append((blockk_ptr, destroy))

        # If necessary, also JIT and register a clean-up kernel for A
        if n % nblock != 0:
            cleank_ptr = create(m, n % nblock, k, lda, ldb, ldc, alpha_ct,
                                beta_ct, a_np.ctypes.data)
            self._kerns.append((cleank_ptr, destroy))
        else:
            cleank_ptr = 0

        # Obtain a pointer to the execute function
        exec_ptr = cast(execute, c_void_p).value

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('par-xsmm').render()

        # Argument types for par_xsmm
        argt = [np.intp, np.intp, np.intp, np.int32, np.int32, np.intp,
                np.intp]

        # Build
        par_xsmm = self._build_kernel('par_xsmm', src, argt)

        class MulKernel(ComputeKernel):
            def run(iself, queue):
                par_xsmm(exec_ptr, blockk_ptr, cleank_ptr, n, nblock, b, out)

        return MulKernel()
