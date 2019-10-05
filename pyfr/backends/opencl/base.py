# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.mpiutil import get_local_rank


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

        # Determine if the device supports double precision arithmetic
        if self.fpdtype == np.float64 and not device.double_fp_config:
            raise ValueError('Device does not support double precision')

        # Create a OpenCL context on this device
        self.ctx = cl.Context([device])

        # Create a queue for initialisation-type operations
        self.qdflt = cl.CommandQueue(self.ctx)

        # Compute the alignment requirement for the context
        self.alignb = device.mem_base_addr_align // 8

        # Compute the SoA size
        self.soasz = 2*self.alignb // np.dtype(self.fpdtype).itemsize

        from pyfr.backends.opencl import (blasext, clblast, gimmik, packing,
                                          provider, types)

        # Register our data types
        self.base_matrix_cls = types.OpenCLMatrixBase
        self.const_matrix_cls = types.OpenCLConstMatrix
        self.matrix_cls = types.OpenCLMatrix
        self.matrix_bank_cls = types.OpenCLMatrixBank
        self.matrix_slice_cls = types.OpenCLMatrixSlice
        self.queue_cls = types.OpenCLQueue
        self.view_cls = types.OpenCLView
        self.xchg_matrix_cls = types.OpenCLXchgMatrix
        self.xchg_view_cls = types.OpenCLXchgView

        # Instantiate the base kernel providers
        kprovs = [provider.OpenCLPointwiseKernelProvider,
                  blasext.OpenCLBlasExtKernels,
                  packing.OpenCLPackingKernels,
                  gimmik.OpenCLGiMMiKKernels,
                  clblast.OpenCLCLBlastKernels]
        self._providers = [k(self) for k in kprovs]

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
        import pyopencl as cl

        # Allocate the device buffer
        buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nbytes)

        # Zero the buffer
        cl.enqueue_copy(self.qdflt, buf, np.zeros(nbytes, dtype=np.uint8))

        return buf
