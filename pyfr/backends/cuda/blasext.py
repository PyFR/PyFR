# -*- coding: utf-8 -*-

import numpy as np
import pycuda.driver as cuda
from pycuda.gpuarray import splay

from pyfr.backends.cuda.provider import CUDAKernelProvider
from pyfr.backends.base import ComputeKernel
from pyfr.nputil import npdtype_to_ctype


class CUDABlasExtKernels(CUDAKernelProvider):
    def axnpby(self, y, *xn):
        if any(y.traits != x.traits for x in xn):
            raise ValueError('Incompatible matrix types')

        nv, cnt = len(xn), y.leaddim*y.nrow

        # Render the kernel template
        tpl = self.backend.lookup.get_template('axnpby')
        src = tpl.render(n=nv, dtype=npdtype_to_ctype(y.dtype))

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
            def run(self, scomp, scopy):
                cuda.memcpy_dtod_async(dst.data, src.data, dst.nbytes,
                                       stream=scomp)

        return CopyKernel()
