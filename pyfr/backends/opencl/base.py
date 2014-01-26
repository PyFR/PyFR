# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.template import DottedTemplateLookup


class OpenCLBackend(BaseBackend):
    name = 'opencl'

    def __init__(self, cfg):
        super(OpenCLBackend, self).__init__(cfg)

        # Create a OpenCL context
        import pyopencl as cl
        self.ctx = cl.create_some_context()

        # Create a queue for initialisation-type operations
        self.qdflt = cl.CommandQueue(self.ctx)

        # Compute the alignment requirement for the context
        self.alignb = self.ctx.devices[0].mem_base_addr_align // 8

        from pyfr.backends.opencl import (blasext, clblas, packing, provider,
                                          types)

        # Register our data types
        self.const_matrix_cls = types.OpenCLConstMatrix
        self.matrix_cls = types.OpenCLMatrix
        self.matrix_bank_cls = types.OpenCLMatrixBank
        self.matrix_rslice_cls = types.OpenCLMatrixRSlice
        self.mpi_matrix_cls = types.OpenCLMPIMatrix
        self.mpi_view_cls = types.OpenCLMPIView
        self.queue_cls = types.OpenCLQueue
        self.view_cls = types.OpenCLView

        # Template lookup
        self.lookup = DottedTemplateLookup('pyfr.backends.opencl.kernels')

        # Instantiate the base kernel providers
        kprovs = [provider.OpenCLPointwiseKernelProvider,
                  blasext.OpenCLBlasExtKernels,
                  packing.OpenCLPackingKernels,
                  clblas.OpenCLClBLASKernels]
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
