# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.mic.provider import MICKernelProvider
from pyfr.backends.base import ComputeKernel


class MICBlasExtKernels(MICKernelProvider):
    def axnpby(self, *arr):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, leaddim, leadsubdim, dtype = arr[0].traits

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(nv=nv)

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32] + [np.intp]*nv + [dtype]*nv)

        # Determine the total element count in the matrices
        cnt = leaddim*nrow

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, *consts):
                args = [x.data for x in arr] + list(consts)
                queue.mic_stream_comp.invoke(kern, cnt, *args)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                queue.mic_stream_comp.transfer_device2device(
                    src.basedata, dst.basedata, dst.nbytes, src.offset,
                    dst.offset
                )

        return CopyKernel()

    def errest(self, x, y, z):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        cnt = x.leaddim*x.nrow
        dtype = x.dtype

        # Allocate space for the return value
        reth = np.zeros(1)
        retd = self.backend.sdflt.bind(reth, update_device=False)

        # Render the reduction kernel template
        src = self.backend.lookup.get_template('errest').render()

        # Build
        rkern = self._build_kernel(
            'errest', src, [np.int32] + [np.intp]*4 + [dtype]*2, restype=dtype
        )

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return float(reth[0])

            def run(self, queue, atol, rtol):
                queue.mic_stream_comp.invoke(
                    rkern, cnt, retd, x.data, y.data, z.data, atol, rtol
                )
                retd.update_host()

        return ErrestKernel()
