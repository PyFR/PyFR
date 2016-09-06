# -*- coding: utf-8 -*-

from pyfr.backends.base import ComputeKernel, NullComputeKernel
from pyfr.backends.base.packing import BasePackingKernels
from pyfr.backends.openmp.provider import OpenMPKernelProvider


class OpenMPPackingKernels(OpenMPKernelProvider, BasePackingKernels):
    def pack(self, mv):
        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, 'iiiPPPP')

        class PackXchgViewKernel(ComputeKernel):
            def run(self, queue):
                kern(v.n, v.nvrow, v.nvcol, v.basedata, v.mapping,
                     v.rstrides or 0, m)

        return PackXchgViewKernel()

    def unpack(self, mv):
        # No-op
        return NullComputeKernel()
