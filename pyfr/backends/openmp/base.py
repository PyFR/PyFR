# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend, blockmats


class OpenMPBackend(BaseBackend):
    name = 'openmp'

    def __init__(self, cfg):
        super(OpenMPBackend, self).__init__(cfg)

        from pyfr.backends.openmp import (blasext, cblas, packing, pointwise,
                                          types)

        # Register our data types
        self.block_diag_matrix_cls = types.OpenMPBlockDiagMatrix
        self.const_matrix_cls = types.OpenMPConstMatrix
        self.matrix_cls = types.OpenMPMatrix
        self.matrix_bank_cls = types.OpenMPMatrixBank
        self.matrix_rslice_cls = types.OpenMPMatrixRSlice
        self.mpi_matrix_cls = types.OpenMPMPIMatrix
        self.mpi_view_cls = types.OpenMPMPIView
        self.queue_cls = types.OpenMPQueue
        self.view_cls = types.OpenMPView

        # Kernel provider classes
        kprovcls = [blockmats.BlockDiagMatrixKernels,
                    pointwise.OpenMPPointwiseKernels,
                    blasext.OpenMPBlasExtKernels,
                    packing.OpenMPPackingKernels,
                    cblas.OpenMPCBLASKernels]
        self._providers = [k(self, cfg) for k in kprovcls]
