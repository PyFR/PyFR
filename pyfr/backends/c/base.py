# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import Backend
from pyfr.backends import blockmats


class CBackend(Backend):
    name = 'c'

    def __init__(self, cfg):
        super(CBackend, self).__init__(cfg)

        from pyfr.backends.c import (blasext, cblas, packing, pointwise, types)

        # Register our data types
        self.const_matrix_cls = types.CConstMatrix
        self.matrix_cls = types.CMatrix
        self.matrix_bank_cls = types.CMatrixBank
        self.matrix_rslice_cls = types.CMatrixRSlice
        self.mpi_matrix_cls = types.CMPIMatrix
        self.mpi_view_cls = types.CMPIView
        self.queue_cls = types.CQueue
        self.view_cls = types.CView

        # Kernel provider classes
        kprovcls = [blockmats.BlockDiagMatrixKernels,
                    pointwise.CPointwiseKernels,
                    blasext.CBlasExtKernels,
                    packing.CPackingKernels,
                    cblas.CBlasKernels]
        self._providers = [k(self, cfg) for k in kprovcls]

        # Numeric data type
        prec = cfg.get('backend', 'precision', 'double')
        if prec not in {'single', 'double'}:
            raise ValueError('C backend precision must be either single or '
                             'double')

        # Convert to a numpy data type
        self.fpdtype = np.dtype(prec).type
