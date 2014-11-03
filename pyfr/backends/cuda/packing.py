# -*- coding: utf-8 -*-

import pycuda.driver as cuda

from pyfr.backends.base import ComputeKernel
from pyfr.backends.base.packing import BasePackingKernels
from pyfr.backends.cuda.provider import CUDAKernelProvider, get_grid_for_block
from pyfr.nputil import npdtype_to_ctype


class CUDAPackingKernels(CUDAKernelProvider, BasePackingKernels):
    def pack(self, mv):
        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        tpl = self.backend.lookup.get_template('pack')
        src = tpl.render(dtype=npdtype_to_ctype(m.dtype))

        # Build
        kern = self._build_kernel('pack_view', src, 'iiiPPPPP')

        # Compute the grid and thread-block size
        block = (128, 1, 1)
        grid = get_grid_for_block(block, v.n)

        # Create a CUDA event
        event = cuda.Event(cuda.event_flags.DISABLE_TIMING)

        class PackXchgViewKernel(ComputeKernel):
            def run(self, queue):
                scomp = queue.cuda_stream_comp
                scopy = queue.cuda_stream_copy

                # Pack
                kern.prepared_async_call(grid, block, scomp, v.n, v.nvrow,
                                         v.nvcol, v.basedata, v.mapping,
                                         v.cstrides or 0, v.rstrides or 0, m)

                # Copy the packed buffer to the host
                event.record(scomp)
                scopy.wait_for_event(event)
                cuda.memcpy_dtoh_async(m.hdata, m.data, scopy)

        return PackXchgViewKernel()

    def unpack(self, mv):
        class UnpackXchgMatrixKernel(ComputeKernel):
            def run(self, queue):
                cuda.memcpy_htod_async(mv.data, mv.hdata,
                                       queue.cuda_stream_comp)

        return UnpackXchgMatrixKernel()
