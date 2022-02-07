# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import Kernel
from pyfr.backends.opencl.provider import OpenCLKernelProvider


class OpenCLPackingKernels(OpenCLKernelProvider):
    def pack(self, mv):
        cl = self.backend.cl

        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, [np.int32]*3 + [np.intp]*4)
        kern.set_args(v.n, v.nvrow, v.nvcol, v.basedata, v.mapping,
                      v.rstrides or 0, m)

        class PackXchgViewKernel(Kernel):
            def run(self, queue):
                # Pack
                kern.exec_async(queue.cmd_q, (v.n,), None)

                # Copy the packed buffer to the host
                cl.memcpy_async(queue.cmd_q, m.hdata, m.data, m.nbytes)

        return PackXchgViewKernel(mats=[mv])

    def unpack(self, mv):
        cl = self.backend.cl

        class UnpackXchgMatrixKernel(Kernel):
            def run(self, queue):
                cl.memcpy_async(queue.cmd_q, mv.data, mv.hdata, mv.nbytes)

        return UnpackXchgMatrixKernel(mats=[mv])
