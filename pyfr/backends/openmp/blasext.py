# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.backends.base import ComputeKernel


class OpenMPBlasExtKernels(OpenMPKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nblocks, nrow, *_, dtype = arr[0].traits
        ncola = arr[0].ioshape[-2]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32]*2 + [np.intp]*nv + [dtype]*nv)

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, *consts):
                kern(nrow, nblocks, *arr, *consts)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        if dst.nbytes >= 2**31:
            raise ValueError('Matrix too large for copy')

        # Render the kernel template
        ksrc = self.backend.lookup.get_template('par-memcpy').render()

        dbbytes, sbbytes = dst.blocksz*dst.itemsize, src.blocksz*src.itemsize
        bnbytes = src.nrow*src.leaddim*src.itemsize
        nblocks = src.nblocks

        # Build the kernel
        kern = self._build_kernel('par_memcpy', ksrc,
                                  [np.intp, np.int32]*2 + [np.int32]*2)

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                kern(dst, dbbytes, src, sbbytes, bnbytes, nblocks)

        return CopyKernel()

    def errest(self, x, y, z, *, norm):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        nblocks, nrow, *_, dtype = x.traits
        ncola = x.ioshape[-2]

        # Render the reduction kernel template
        src = self.backend.lookup.get_template('errest').render(norm=norm,
                                                                ncola=ncola)

        # Array for the error estimate
        error = np.zeros(ncola, dtype=dtype)

        # Build
        rkern = self._build_kernel(
            'errest', src, [np.int32]*2 + [np.intp]*4 + [dtype]*2
        )

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return error

            def run(self, queue, atol, rtol):
                rkern(nrow, nblocks, error.ctypes.data, x, y, z, atol, rtol)

        return ErrestKernel()
