# -*- coding: utf-8 -*-

from pyfr.backends.base import ComputeKernel
from pyfr.backends.base.packing import BasePackingKernels
from pyfr.backends.mic.provider import MICKernelProvider


class MICPackingKernels(MICKernelProvider, BasePackingKernels):
    def pack(self, mv):
        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, 'iiiPPPP')

        class PackXchgViewKernel(ComputeKernel):
            def run(self, queue):
                # Kernel arguments
                args = [v.n, v.nvrow, v.nvcol, v.basedata.dev_ptr,
                        v.mapping, v.rstrides, m]
                args = [getattr(arg, 'data', arg) for arg in args]

                # Pack
                queue.mic_stream_comp.invoke(kern, *args)

                # Copy the packed buffer to the host
                queue.mic_stream_comp.transfer_device2host(
                    m.basedata, m.hdata.ctypes.data, m.nbytes,
                    offset_device=m.offset
                )

        return PackXchgViewKernel()

    def unpack(self, mv):
        class UnpackXchgMatrixKernel(ComputeKernel):
            def run(self, queue):
                queue.mic_stream_comp.transfer_host2device(
                    mv.hdata.ctypes.data, mv.basedata, mv.nbytes,
                    offset_device=mv.offset
                )

        return UnpackXchgMatrixKernel()
