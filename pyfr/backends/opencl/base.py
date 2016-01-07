# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.mpiutil import get_local_rank
from pyfr.template import DottedTemplateLookup


class OpenCLBackend(BaseBackend):
    name = 'opencl'

    def __init__(self, cfg):
        super().__init__(cfg)

        import pyopencl as cl

        # Get the platform/device info from the config file
        platid = cfg.get('backend-opencl', 'platform-id', '0').lower()
        devid = cfg.get('backend-opencl', 'device-id', 'local-rank').lower()
        devtype = cfg.get('backend-opencl', 'device-type', 'all').upper()

        # Handle the local-rank case
        if devid == 'local-rank':
            devid = str(get_local_rank())

        # Map the device type to the corresponding PyOpenCL constant
        devtype = getattr(cl.device_type, devtype)

        # Determine the OpenCL platform to use
        for i, platform in enumerate(cl.get_platforms()):
            if platid == str(i) or platid == platform.name.lower():
                break
        else:
            raise ValueError('No suitable OpenCL platform found')

        # Determine the OpenCL device to use
        for i, device in enumerate(platform.get_devices(devtype)):
            if devid == str(i) or devid == device.name.lower():
                break
        else:
            raise ValueError('No suitable OpenCL device found')

        # Create a OpenCL context on this device
        self.ctx = cl.Context([device])

        # Create a queue for initialisation-type operations
        self.qdflt = cl.CommandQueue(self.ctx)

        # Compute the alignment requirement for the context
        self.alignb = device.mem_base_addr_align // 8

        from pyfr.backends.opencl import (blasext, clblas, gimmik, packing,
                                          provider, types)

        # Register our data types
        self.base_matrix_cls = types.OpenCLMatrixBase
        self.const_matrix_cls = types.OpenCLConstMatrix
        self.matrix_cls = types.OpenCLMatrix
        self.matrix_bank_cls = types.OpenCLMatrixBank
        self.matrix_rslice_cls = types.OpenCLMatrixRSlice
        self.queue_cls = types.OpenCLQueue
        self.view_cls = types.OpenCLView
        self.xchg_matrix_cls = types.OpenCLXchgMatrix
        self.xchg_view_cls = types.OpenCLXchgView

        # Template lookup
        self.lookup = DottedTemplateLookup(
            'pyfr.backends.opencl.kernels',
            fpdtype=self.fpdtype, alignb=self.alignb
        )

        # Instantiate the base kernel providers
        kprovs = [provider.OpenCLPointwiseKernelProvider,
                  blasext.OpenCLBlasExtKernels,
                  packing.OpenCLPackingKernels,
                  gimmik.OpenCLGiMMiKKernels,
                  clblas.OpenCLClBLASKernels]
        self._providers = [k(self) for k in kprovs]

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
        import pyopencl as cl

        # Allocate the device buffer; note here that we over allocate
        # by a byte.  This is needed to work around some issues in
        # related to the construction of sub buffers.  (For which the
        # solution is to increase the size of the region by one byte;
        # hence requiring an extra byte of allocation.)
        buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nbytes + 1)

        # Zero the buffer
        cl.enqueue_copy(self.qdflt, buf, np.zeros(nbytes + 1, dtype=np.uint8))

        return buf
