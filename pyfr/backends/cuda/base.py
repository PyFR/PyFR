# -*- coding: utf-8 -*-

from pyfr.backends.base import BaseBackend
from pyfr.template import DottedTemplateLookup


class CUDABackend(BaseBackend):
    name = 'cuda'

    def __init__(self, cfg):
        super(CUDABackend, self).__init__(cfg)

        # Create a CUDA context
        from pycuda.autoinit import context, device

        # Get the alignment requirements for the device
        self.alignb = device.texture_alignment

        # Some CUDA devices share L1 cache and shared memory; on these
        # devices CUDA allows us to specify a preference between L1
        # cache and shared memory.  For the sake of CUBLAS (which
        # benefits greatly from more shared memory but fails to
        # declare its preference) we set the global default to
        # PREFER_SHARED.
        from pycuda.driver import func_cache
        context.set_cache_config(func_cache.PREFER_SHARED)

        from pyfr.backends.cuda import (blasext, cublas, packing, provider,
                                        types)

        # Register our data types
        self.const_matrix_cls = types.CUDAConstMatrix
        self.matrix_cls = types.CUDAMatrix
        self.matrix_bank_cls = types.CUDAMatrixBank
        self.matrix_rslice_cls = types.CUDAMatrixRSlice
        self.mpi_matrix_cls = types.CUDAMPIMatrix
        self.mpi_view_cls = types.CUDAMPIView
        self.queue_cls = types.CUDAQueue
        self.view_cls = types.CUDAView

        # Template lookup
        self.lookup = DottedTemplateLookup('pyfr.backends.cuda.kernels')

        # Instantiate the base kernel providers
        kprovs = [provider.CUDAPointwiseKernelProvider,
                  blasext.CUDABlasExtKernels,
                  packing.CUDAPackingKernels,
                  cublas.CUDACUBLASKernels]
        self._providers = [k(self) for k in kprovs]

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
        import pycuda.driver as cuda

        # Allocate
        data = cuda.mem_alloc(nbytes)

        # Zero
        cuda.memset_d32(data, 0, nbytes // 4)

        return data
