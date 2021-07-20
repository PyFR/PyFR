# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import ComputeKernel, NullComputeKernel
from pyfr.backends.base.packing import BasePackingKernels
from pyfr.backends.hip.provider import HIPKernelProvider, get_grid_for_block


class HIPPackingKernels(HIPKernelProvider, BasePackingKernels):
    def pack(self, mv):
        hip = self.backend.hip

        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Compute the grid and thread-block size
        block = (128, 1, 1)
        grid = get_grid_for_block(block, v.n)

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render(blocksz=block[0])

        # Build
        kern = self._build_kernel('pack_view', src, [np.int32]*3 + [np.intp]*4)

        # If MPI is HIP aware then we just need to pack the buffer
        if self.backend.mpitype == 'hip-aware':
            class PackXchgViewKernel(ComputeKernel):
                def run(self, queue):
                    # Pack
                    kern.exec_async(
                        grid, block, queue.stream_comp, v.n, v.nvrow, v.nvcol,
                        v.basedata, v.mapping, v.rstrides or 0, m
                    )
        # Otherwise, we need to both pack the buffer and copy it back
        else:
            # Create a HIP event
            event = hip.create_event()

            class PackXchgViewKernel(ComputeKernel):
                def run(self, queue):
                    # Pack
                    kern.exec_async(
                        grid, block, queue.stream_comp, v.n, v.nvrow, v.nvcol,
                        v.basedata, v.mapping, v.rstrides or 0, m
                    )

                    # Copy the packed buffer to the host
                    event.record(queue.stream_comp)
                    queue.stream_copy.wait_for_event(event)
                    hip.memcpy_async(m.hdata, m.data, m.nbytes,
                                     queue.stream_copy)

        return PackXchgViewKernel()

    def unpack(self, mv):
        hip = self.backend.hip

        if self.backend.mpitype == 'hip-aware':
            return NullComputeKernel()
        else:
            class UnpackXchgMatrixKernel(ComputeKernel):
                def run(self, queue):
                    hip.memcpy_async(mv.data, mv.hdata, mv.nbytes,
                                     queue.stream_comp)

            return UnpackXchgMatrixKernel()
