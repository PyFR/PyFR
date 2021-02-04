# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import ComputeKernel, NullComputeKernel
from pyfr.backends.base.packing import BasePackingKernels
from pyfr.backends.cuda.provider import CUDAKernelProvider, get_grid_for_block


class CUDAPackingKernels(CUDAKernelProvider, BasePackingKernels):
    def pack(self, mv):
        cuda = self.backend.cuda

        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, [np.int32]*3 + [np.intp]*4)

        # Compute the grid and thread-block size
        block = (128, 1, 1)
        grid = get_grid_for_block(block, v.n)

        # If MPI is CUDA aware then we just need to pack the buffer
        if self.backend.mpitype == 'cuda-aware':
            class PackXchgViewKernel(ComputeKernel):
                def run(self, queue):
                    scomp = queue.cuda_stream_comp

                    # Pack
                    kern.exec_async(
                        grid, block, scomp, v.n, v.nvrow, v.nvcol, v.basedata,
                        v.mapping, v.rstrides or 0, m
                    )
        # Otherwise, we need to both pack the buffer and copy it back
        else:
            # Create a CUDA event
            event = cuda.create_event()

            class PackXchgViewKernel(ComputeKernel):
                def run(self, queue):
                    scomp = queue.cuda_stream_comp
                    scopy = queue.cuda_stream_copy

                    # Pack
                    kern.exec_async(
                        grid, block, scomp, v.n, v.nvrow, v.nvcol, v.basedata,
                        v.mapping, v.rstrides or 0, m
                    )

                    # Copy the packed buffer to the host
                    event.record(scomp)
                    scopy.wait_for_event(event)
                    cuda.memcpy_async(m.hdata, m.data, m.nbytes, scopy)

        return PackXchgViewKernel()

    def unpack(self, mv):
        cuda = self.backend.cuda

        if self.backend.mpitype == 'cuda-aware':
            return NullComputeKernel()
        else:
            class UnpackXchgMatrixKernel(ComputeKernel):
                def run(self, queue):
                    cuda.memcpy_async(mv.data, mv.hdata, mv.nbytes,
                                      queue.cuda_stream_comp)

            return UnpackXchgMatrixKernel()
