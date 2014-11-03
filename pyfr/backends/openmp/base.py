# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.template import DottedTemplateLookup


class OpenMPBackend(BaseBackend):
    name = 'openmp'

    def __init__(self, cfg):
        super(OpenMPBackend, self).__init__(cfg)

        # Take the alignment requirement to be 32-bytes
        self.alignb = 32

        from pyfr.backends.openmp import (blasext, cblas, packing, provider,
                                          types)

        # Register our data types
        self.base_matrix_cls = types.OpenMPMatrixBase
        self.const_matrix_cls = types.OpenMPConstMatrix
        self.matrix_cls = types.OpenMPMatrix
        self.matrix_bank_cls = types.OpenMPMatrixBank
        self.matrix_rslice_cls = types.OpenMPMatrixRSlice
        self.queue_cls = types.OpenMPQueue
        self.view_cls = types.OpenMPView
        self.xchg_matrix_cls = types.OpenMPXchgMatrix
        self.xchg_view_cls = types.OpenMPXchgView

        # Template lookup
        self.lookup = DottedTemplateLookup('pyfr.backends.openmp.kernels')

        # Kernel provider classes
        kprovcls = [provider.OpenMPPointwiseKernelProvider,
                    blasext.OpenMPBlasExtKernels,
                    packing.OpenMPPackingKernels,
                    cblas.OpenMPCBLASKernels]
        self._providers = [k(self) for k in kprovcls]

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
            data = np.zeros(nbytes + self.alignb, dtype=np.uint8)
            offset = -data.ctypes.data % self.alignb

            return data[offset:nbytes + offset]
