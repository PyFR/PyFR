# -*- coding: utf-8 -*-

from pyfr.backends.base import Kernel, NullKernel
from pyfr.backends.openmp.provider import OpenMPKernelProvider


class OpenMPPackingKernels(OpenMPKernelProvider):
    def pack(self, mv):
        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render(nrv=v.nvrow,
                                                              ncv=v.nvcol)

        # Build
        kern = self._build_kernel('pack_view', src, 'iPPPP')
        kern.set_args(v.n, v.basedata, v.mapping, v.rstrides or 0, m)

        class PackXchgViewKernel(Kernel):
            def run(self, queue):
                kern()

        return PackXchgViewKernel(mats=[mv])

    def unpack(self, mv):
        return NullKernel()
