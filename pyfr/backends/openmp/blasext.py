# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.backends.base import ComputeKernel


class OpenMPBlasExtKernels(OpenMPKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, ncol, ldim, dtype = arr[0].traits
        ncola, ncolb = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32]*3 + [np.intp]*nv + [dtype]*nv)

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, *consts):
                args = list(arr) + list(consts)
                kern(nrow, ncolb, ldim, *args)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        if dst.nbytes >= 2**31:
            raise ValueError('Matrix too large for copy')

        # Render the kernel template
        ksrc = self.backend.lookup.get_template('par-memcpy').render()

        # Build the kernel
        kern = self._build_kernel('par_memcpy', ksrc,
                                  [np.intp, np.intp, np.int32])

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                kern(dst, src, dst.nbytes)

        return CopyKernel()

    def errest(self, x, y, z, *, norm):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        nrow, ncol, ldim, dtype = x.traits
        ncola, ncolb = x.ioshape[1:]

        # Render the reduction kernel template
        src = self.backend.lookup.get_template('errest').render(norm=norm,
                                                                ncola=ncola)

        # Array for the error estimate
        error = np.zeros(ncola, dtype=dtype)

        # Build
        rkern = self._build_kernel(
            'errest', src, [np.int32]*3 + [np.intp]*4 + [dtype]*2,
            restype=dtype
        )

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return error

            def run(self, queue, atol, rtol):
                rkern(nrow, ncolb, ldim, error.ctypes.data,
                      x, y, z, atol, rtol)

        return ErrestKernel()
