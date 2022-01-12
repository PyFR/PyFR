# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.cuda.provider import CUDAKernelProvider, get_grid_for_block
from pyfr.backends.base import Kernel


class CUDABlasExtKernels(CUDAKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, ncol, ldim, dtype = arr[0].traits[1:]
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

        class AxnpbyKernel(Kernel):
            def run(self, queue, *consts):
                kern.exec_async(grid, block, queue.stream, nrow, ncolb, ldim,
                                *arr, *consts)

        return AxnpbyKernel()

    def copy(self, dst, src):
        cuda = self.backend.cuda

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(Kernel):
            def run(self, queue):
                cuda.memcpy(dst, src, dst.nbytes, queue.stream)

        return CopyKernel()

    def reduction(self, *rs, method, norm, dt_mat=None):
        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        cuda = self.backend.cuda
        nrow, ncol, ldim, dtype = rs[0].traits[1:]
        ncola, ncolb = rs[0].ioshape[1:]

        # Reduction block dimensions
        block = (128, 1, 1)

        # Determine the grid size
        grid = get_grid_for_block(block, ncolb, ncola)

        # Empty result buffer on the device
        reduced_dev = cuda.mem_alloc(ncola*grid[0]*rs[0].itemsize)

        # Empty result buffer on the host
        reduced_host = cuda.pagelocked_empty((ncola, grid[0]), dtype)

        tplargs = dict(norm=norm, sharesz=block[0], method=method)

        if method == 'resid':
            tplargs['dt_type'] = 'matrix' if dt_mat else 'scalar'

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        regs = list(rs) + [dt_mat] if dt_mat else rs

        # Argument types for reduction kernel
        if method == 'errest':
            argt = [np.int32]*3 + [np.intp]*4 + [dtype]*2
        elif method == 'resid' and dt_mat:
            argt = [np.int32]*3 + [np.intp]*4 + [dtype]
        else:
            argt = [np.int32]*3 + [np.intp]*3 + [dtype]

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)

        # Norm type
        reducer = np.max if norm == 'uniform' else np.sum

        class ReductionKernel(Kernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            def run(self, queue, *facs):
                rkern.exec_async(grid, block, queue.stream,
                                 nrow, ncolb, ldim, reduced_dev, *regs, *facs)
                cuda.memcpy(reduced_host, reduced_dev, reduced_dev.nbytes,
                            queue.stream)

        return ReductionKernel()
