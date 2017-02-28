# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend


class OpenMPBackend(BaseBackend):
    name = 'openmp'

    def __init__(self, cfg):
        super().__init__(cfg)

        # Take the alignment requirement to be 64-bytes
        self.alignb = 64

        # Compute the SoA size
        self.soasz = self.alignb // np.dtype(self.fpdtype).itemsize

        from pyfr.backends.openmp import (blasext, cblas, gimmik, packing,
                                          provider, types, xsmm)

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

        # Instantiate mandatory kernel provider classes
        kprovcls = [provider.OpenMPPointwiseKernelProvider,
                    blasext.OpenMPBlasExtKernels,
                    packing.OpenMPPackingKernels,
                    gimmik.OpenMPGiMMiKKernels]
        self._providers = [k(self) for k in kprovcls]

        # Instantiate optional kernel provider classes
        for k in [xsmm.OpenMPXSMMKernels, cblas.OpenMPCBLASKernels]:
            try:
                self._providers.append(k(self))
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                pass

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
        data = np.zeros(nbytes + self.alignb, dtype=np.uint8)
        offset = -data.ctypes.data % self.alignb

        return data[offset:nbytes + offset]
