# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
from pyopencl.array import Array
from pyopencl.reduction import ReductionKernel

from pyfr.backends.opencl.provider import OpenCLKernelProvider
from pyfr.backends.base import ComputeKernel
from pyfr.nputil import npdtype_to_ctype


class OpenCLBlasExtKernels(OpenCLKernelProvider):
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
                args = [x.data for x in arr] + list(consts)
                kern(queue.cl_queue_comp, (ncolb, nrow), None, nrow, ncolb,
                     ldim, *args)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                cl.enqueue_copy(queue.cl_queue_comp, dst.data, src.data)

        return CopyKernel()

    def errest(self, x, y, z, *, norm):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        nrow, ncol, ldim, dtype = x.traits
        ncola, ncolb = x.ioshape[1:]

        # Reduction workgroup dimensions
        ls = (128,)
        gs = (ncolb - ncolb % -ls[0],)

        # Empty result buffer on host with (nvars, ngroups)
        err_host = np.empty((ncola, gs[0] // ls[0]), dtype)

        # Device memory allocation
        err_dev = cl.Buffer(self.backend.ctx, cl.mem_flags.READ_WRITE,
                            err_host.nbytes)

        # Get the kernel template
        src = self.backend.lookup.get_template('errest').render(
            norm=norm, ncola=ncola, sharesz=ls[0]
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
                rkern(queue.cl_queue_comp, gs, ls, nrow, ncolb, ldim, err_dev,
                      x.data, y.data, z.data, atol, rtol)
                cl.enqueue_copy(queue.cl_queue_comp, err_host, err_dev,
                                is_blocking=False)

        return ErrestKernel()
