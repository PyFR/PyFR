# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.backends.base import ComputeKernel


class OpenMPBlasExtKernels(OpenMPKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        ncola, ncolb = arr[0].datashape[1:]
        nrow, ldim, lsdim, dtype = arr[0].traits

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32]*4 + [np.intp]*nv + [dtype]*nv)

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, *consts):
                args = list(arr) + list(consts)
                kern(nrow, ncolb, ldim, lsdim, *args)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                dst.data[:] = src.data.reshape(dst.data.shape)

        return CopyKernel()

    def errest(self, x, y, z):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        cnt = x.leaddim*x.nrow
        dtype = x.dtype

        # Render the reduction kernel template
        src = self.backend.lookup.get_template('errest').render()

        # Build
        rkern = self._build_kernel(
            'errest', src, [np.int32] + [np.intp]*3 + [dtype]*2, restype=dtype
        )

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return self._retval

            def run(self, queue, atol, rtol):
                self._retval = rkern(cnt, x, y, z, atol, rtol)

        return ErrestKernel()
