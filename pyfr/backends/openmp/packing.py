import numpy as np

from pyfr.backends.base import NullKernel
from pyfr.backends.openmp.provider import OpenMPKernel, OpenMPKernelProvider


class OpenMPPackingKernels(OpenMPKernelProvider):
    def _packing_kern(self, type, v, xm):
        # Render the kernel template
        src = self.backend.lookup.get_template('packing').render(nrv=v.nvrow,
                                                                 ncv=v.nvcol)

        # Build
        kern = self._build_kernel(f'{type}_view', src,
                                  [self.backend.ixdtype] + [np.uintp]*4)
        kern.set_args(v.n, v.basedata, v.mapping, v.rstrides or 0, xm)

        return kern

    def pack(self, xmv):
        if isinstance(xmv, self.backend.xchg_matrix_cls):
            return NullKernel()
        else:
            kern = self._packing_kern('pack', xmv.view, xmv.xchgmat)

            return OpenMPKernel(mats=[xmv], kernel=kern)

    def unpack(self, xmv):
        if isinstance(xmv, self.backend.xchg_matrix_cls):
            return NullKernel()
        else:
            kern = self._packing_kern('unpack', xmv.view, xmv.xchgmat)

            return OpenMPKernel(mats=[xmv], kernel=kern)
