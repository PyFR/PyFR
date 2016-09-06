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
        nrow, ldim, dtype = arr[0].traits
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

        cnt = x.leaddim*x.nrow
        dtype = x.dtype

        # Norm type
        reduce_expr = 'a + b' if norm == 'l2' else 'max(a, b)'

        # Build the reduction kernel
        rkern = ReductionKernel(
            self.backend.ctx, dtype, neutral='0', reduce_expr=reduce_expr,
            map_expr='pow(x[i]/(atol + rtol*max(fabs(y[i]), fabs(z[i]))), 2)',
            arguments='__global {0}* x, __global {0}* y, __global {0}* z, '
                      '{0} atol, {0} rtol'.format(npdtype_to_ctype(dtype))
        )

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return self._retarr.get()

            def run(self, queue, atol, rtol):
                qcomp = queue.cl_queue_comp

                xarr = Array(qcomp, cnt, dtype, data=x.data)
                yarr = Array(qcomp, cnt, dtype, data=y.data)
                zarr = Array(qcomp, cnt, dtype, data=z.data)

                self._retarr = rkern(xarr, yarr, zarr, atol, rtol,
                                     queue=qcomp)

        return ErrestKernel()
