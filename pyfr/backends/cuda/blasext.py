# -*- coding: utf-8 -*-

import numpy as np
import pycuda.driver as cuda
from pycuda.gpuarray import GPUArray, splay
from pycuda.reduction import ReductionKernel

from pyfr.backends.cuda.provider import CUDAKernelProvider
from pyfr.backends.base import ComputeKernel
from pyfr.nputil import npdtype_to_ctype


class CUDABlasExtKernels(CUDAKernelProvider):
    def axnpby(self, y, *xn):
        if any(y.traits != x.traits for x in xn):
            raise ValueError('Incompatible matrix types')

        nv, cnt = len(xn), y.leaddim*y.nrow

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(n=nv)

        # Build
        kern = self._build_kernel('axnpby', src,
                                  [np.int32] + [np.intp, y.dtype]*(1 + nv))

        # Compute a suitable block and grid
        grid, block = splay(cnt)

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, beta, *alphan):
                args = [i for axn in zip(xn, alphan) for i in axn]
                kern.prepared_async_call(grid, block, queue.cuda_stream_comp,
                                         cnt, y, beta, *args)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                cuda.memcpy_dtod_async(dst.data, src.data, dst.nbytes,
                                       stream=queue.cuda_stream_comp)

        return CopyKernel()

    def errest(self, x, y, z):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        # Wrap
        xarr = GPUArray(x.leaddim*x.nrow, x.dtype, gpudata=x)
        yarr = GPUArray(y.leaddim*y.nrow, y.dtype, gpudata=y)
        zarr = GPUArray(z.leaddim*z.nrow, z.dtype, gpudata=z)

        # Build the reduction kernel
        rkern = ReductionKernel(
            x.dtype, neutral='0', reduce_expr='a + b',
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
