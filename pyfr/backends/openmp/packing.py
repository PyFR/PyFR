import numpy as np

from pyfr.backends.base import NullKernel
from pyfr.backends.openmp.provider import OpenMPKernel, OpenMPKernelProvider


class OpenMPPackingKernels(OpenMPKernelProvider):
    def pack(self, mv):
        ixdtype = self.backend.ixdtype

        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render(nrv=v.nvrow,
                                                              ncv=v.nvcol)

        # Build
        kern = self._build_kernel('pack_view', src, [ixdtype] + [np.uintp]*4)
        kern.set_args(v.n, v.basedata, v.mapping, v.rstrides or 0, m)

        return OpenMPKernel(mats=[mv], kernel=kern)

    def unpack(self, mv):
        return NullKernel()
