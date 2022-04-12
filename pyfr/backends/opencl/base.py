# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.mpiutil import get_local_rank


class OpenCLBackend(BaseBackend):
    name = 'opencl'
    blocks = False

    def __init__(self, cfg):
        super().__init__(cfg)

        from pyfr.backends.opencl.driver import OpenCL

        # Load and wrap OpenCL
        self.cl = OpenCL()

        # Get the platform/device info from the config file
        platid = cfg.get('backend-opencl', 'platform-id', '0').lower()
        devid = cfg.get('backend-opencl', 'device-id', 'local-rank').lower()
        devtype = cfg.get('backend-opencl', 'device-type', 'all').upper()

        # Handle the local-rank case
        if devid == 'local-rank':
            devid = str(get_local_rank())

        # Determine the OpenCL platform to use
        for i, platform in enumerate(self.cl.get_platforms()):
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
        if self.fpdtype == np.float64 and not device.has_fp64:
            raise ValueError('Device does not support double precision')

        # Set the device
        self.cl.set_device(device)

        # Compute the alignment requirement for the context
        self.alignb = device.mem_align

        # Compute the SoA size
        self.soasz = 2*self.alignb // np.dtype(self.fpdtype).itemsize
        self.csubsz = self.soasz

        from pyfr.backends.opencl import (blasext, clblast, gimmik, packing,
                                          provider, types)

        # Register our data types
        self.base_matrix_cls = types.OpenCLMatrixBase
        self.const_matrix_cls = types.OpenCLConstMatrix
        self.matrix_cls = types.OpenCLMatrix
        self.matrix_slice_cls = types.OpenCLMatrixSlice
        self.queue_cls = types.OpenCLQueue
        self.view_cls = types.OpenCLView
        self.xchg_matrix_cls = types.OpenCLXchgMatrix
        self.xchg_view_cls = types.OpenCLXchgView

        # Instantiate the base kernel providers
        kprovs = [provider.OpenCLPointwiseKernelProvider,
                  blasext.OpenCLBlasExtKernels,
                  packing.OpenCLPackingKernels,
                  gimmik.OpenCLGiMMiKKernels]
        self._providers = [k(self) for k in kprovs]

        # Load CLBlast if available
        try:
            self._providers.append(clblast.OpenCLCLBlastKernels(self))
        except OSError:
            pass

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
        # Allocate the device buffer
        buf = self.cl.mem_alloc(nbytes)

        # Zero the buffer
        self.cl.zero(buf, 0, nbytes)

        return buf
