# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.mpiutil import get_local_rank


class MICBackend(BaseBackend):
    name = 'mic'

    def __init__(self, cfg):
        super().__init__(cfg)

        import pymic as mic

        # Get the device ID to use
        devid = cfg.get('backend-mic', 'device-id', 'local-rank')

        # Handle the local-rank case
        if devid == 'local-rank':
            devid = str(get_local_rank())

        # Get a handle to the desired device
        self.dev = mic.devices[int(devid)]

        # Default stream
        self.sdflt = self.dev.get_default_stream()

        # Take the alignment requirement to be 64-bytes
        self.alignb = 64

        # Compute the SoA size
        self.soasz = self.alignb // np.dtype(self.fpdtype).itemsize

        from pyfr.backends.mic import (blasext, cblas, packing, provider,
                                       types)

        # Register our data types
        self.base_matrix_cls = types.MICMatrixBase
        self.const_matrix_cls = types.MICConstMatrix
        self.matrix_cls = types.MICMatrix
        self.matrix_bank_cls = types.MICMatrixBank
        self.matrix_rslice_cls = types.MICMatrixRSlice
        self.queue_cls = types.MICQueue
        self.view_cls = types.MICView
        self.xchg_matrix_cls = types.MICXchgMatrix
        self.xchg_view_cls = types.MICXchgView

        # Kernel provider classes
        kprovcls = [provider.MICPointwiseKernelProvider,
                    blasext.MICBlasExtKernels,
                    packing.MICPackingKernels,
                    cblas.MICCBLASKernels]
        self._providers = [k(self) for k in kprovcls]

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
        stream = self.sdflt

        # Allocate an empty buffer on the device
        buf = stream.allocate_device_memory(nbytes)

        # Attach the raw device pointer
        buf.dev_ptr = stream.translate_device_pointer(buf)

        # Zero the buffer
        zeros = np.zeros(nbytes, dtype=np.uint8)
        stream.transfer_host2device(zeros.ctypes.data, buf, nbytes)
        stream.sync()

        return buf
