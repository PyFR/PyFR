# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

from pyfr.backends.base import ComputeKernel
from pyfr.backends.base.packing import BasePackingKernels
from pyfr.backends.opencl.provider import OpenCLKernelProvider


class OpenCLPackingKernels(OpenCLKernelProvider, BasePackingKernels):
    def pack(self, mv):
        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, [np.int32]*3 + [np.intp]*4)

        class PackXchgViewKernel(ComputeKernel):
            def run(self, queue):
                # Kernel arguments
                args = [v.n, v.nvrow, v.nvcol, v.basedata, v.mapping,
                        v.rstrides, m]
                args = [getattr(arg, 'data', arg) for arg in args]

                # Pack
                pevent = kern(queue.cl_queue_comp, (v.n,), None, *args)

                # Copy the packed buffer to the host
                cevent = cl.enqueue_copy(queue.cl_queue_copy, m.hdata, m.data,
                                         is_blocking=False, wait_for=[pevent])
                queue.copy_events.append(cevent)

        return PackXchgViewKernel()

    def unpack(self, mv):
        class UnpackXchgMatrixKernel(ComputeKernel):
            def run(self, queue):
                cevent = cl.enqueue_copy(queue.cl_queue_comp, mv.data,
                                         mv.hdata, is_blocking=False)
                queue.copy_events.append(cevent)

        return UnpackXchgMatrixKernel()
