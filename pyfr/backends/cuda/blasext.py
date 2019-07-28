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
        nrow, ncol, ldim, dtype = arr[0].traits
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

        nrow, ncol, ldim, dtype = x.traits
        ncola, ncolb = x.ioshape[1:]

        # Reduction block dimensions
        block = (128, 1, 1)

        # Determine the grid size
        grid = get_grid_for_block(block, ncolb)

        # Empty result buffer on host with shape (nvars, nblocks)
        err_host = cuda.pagelocked_empty((ncola, grid[0]), dtype, 'C')

        # Device memory allocation
        err_dev = cuda.mem_alloc(err_host.nbytes)

        # Get the kernel template
        src = self.backend.lookup.get_template('errest').render(
            norm=norm, ncola=ncola, sharesz=block[0]
        )

        # Build the reduction kernel
        rkern = self._build_kernel(
            'errest', src, [np.int32]*3 + [np.intp]*4 + [dtype]*2
        )

        # Norm type
        reducer = np.max if norm == 'uniform' else np.sum

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return reducer(err_host, axis=1)

            def run(self, queue, atol, rtol):
                rkern.prepared_async_call(grid, block, queue.cuda_stream_comp,
                                          nrow, ncolb, ldim, err_dev, x, y, z,
                                          atol, rtol)
                cuda.memcpy_dtoh_async(err_host, err_dev,
                                       queue.cuda_stream_comp)

        return ErrestKernel()
