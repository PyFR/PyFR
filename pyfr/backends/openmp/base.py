# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.backends.openmp.compiler import OpenMPCompiler


class OpenMPBackend(BaseBackend):
    name = 'openmp'
    blocks = True

    def __init__(self, cfg):
        super().__init__(cfg)

        # Take the default alignment requirement to be 64-bytes
        self.alignb = cfg.getint('backend-openmp', 'alignb', 64)

        if self.alignb < 32 or (self.alignb & (self.alignb - 1)):
            raise ValueError('Alignment must be a power of 2 and >= 32')

        # Compute the SoA and AoSoA size
        self.soasz = self.alignb // np.dtype(self.fpdtype).itemsize
        self.csubsz = self.soasz*cfg.getint('backend-openmp', 'n-soa', 1)

        # C source compiler
        self.compiler = OpenMPCompiler(cfg)

        from pyfr.backends.openmp import (blasext, packing, provider, types,
                                          xsmm)

        # Register our data types
        self.base_matrix_cls = types.OpenMPMatrixBase
        self.const_matrix_cls = types.OpenMPConstMatrix
        self.matrix_cls = types.OpenMPMatrix
        self.matrix_slice_cls = types.OpenMPMatrixSlice
        self.queue_cls = types.OpenMPQueue
        self.view_cls = types.OpenMPView
        self.xchg_matrix_cls = types.OpenMPXchgMatrix
        self.xchg_view_cls = types.OpenMPXchgView

        # Instantiate mandatory kernel provider classes
        kprovcls = [provider.OpenMPPointwiseKernelProvider,
                    blasext.OpenMPBlasExtKernels,
                    packing.OpenMPPackingKernels,
                    xsmm.OpenMPXSMMKernels]
        self._providers = [k(self) for k in kprovcls]

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
        data = np.zeros(nbytes + self.alignb, dtype=np.uint8)
        offset = -data.ctypes.data % self.alignb

        return data[offset:nbytes + offset]
