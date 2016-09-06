# -*- coding: utf-8 -*-

import numpy as np
import pycuda.driver as cuda
from pycuda.gpuarray import GPUArray
from pycuda.reduction import ReductionKernel

from pyfr.backends.cuda.provider import CUDAKernelProvider, get_grid_for_block
from pyfr.backends.base import ComputeKernel
from pyfr.nputil import npdtype_to_ctype


class CUDABlasExtKernels(CUDAKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, ldim, dtype = arr[0].traits
        ncola, ncolb = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32]*3 + [np.intp]*nv + [dtype]*nv)

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow)

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, *consts):
                args = list(arr) + list(consts)

                kern.prepared_async_call(grid, block, queue.cuda_stream_comp,
                                         nrow, ncolb, ldim, *args)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                cuda.memcpy_dtod_async(dst.data, src.data, dst.nbytes,
                                       stream=queue.cuda_stream_comp)

        return CopyKernel()

    def errest(self, x, y, z, *, norm):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        # Wrap
        xarr = GPUArray(x.leaddim*x.nrow, x.dtype, gpudata=x)
        yarr = GPUArray(y.leaddim*y.nrow, y.dtype, gpudata=y)
        zarr = GPUArray(z.leaddim*z.nrow, z.dtype, gpudata=z)

        # Norm type
        reduce_expr = 'a + b' if norm == 'l2' else 'max(a, b)'

        # Build the reduction kernel
        rkern = ReductionKernel(
            x.dtype, neutral='0', reduce_expr=reduce_expr,
            map_expr='pow(x[i]/(atol + rtol*max(fabs(y[i]), fabs(z[i]))), 2)',
            arguments='{0}* x, {0}* y, {0}* z, {0} atol, {0} rtol'
                      .format(npdtype_to_ctype(x.dtype))
        )

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return self._retarr.get()

            def run(self, queue, atol, rtol):
                self._retarr = rkern(xarr, yarr, zarr, atol, rtol,
                                     stream=queue.cuda_stream_comp)

        return ErrestKernel()
