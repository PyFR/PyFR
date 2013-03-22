# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import Backend
from pyfr.backends import blockmats


class CudaBackend(Backend):
    name = 'CUDA'

    def __init__(self, cfg):
        super(CudaBackend, self).__init__(cfg)

        # Create a CUDA context
        from pycuda.autoinit import context as cuda_ctx

        # Some CUDA devices share L1 cache and shared memory; on these
        # devices CUDA allows us to specify a preference between L1
        # cache and shared memory.  For the sake of CUBLAS (which
        # benefits greatly from more shared memory but fails to
        # declare its preference) we set the global default to
        # PREFER_SHARED.
        from pycuda.driver import func_cache
        cuda_ctx.set_cache_config(func_cache.PREFER_SHARED)

        # For introspection to work it must always be possible to
        # import the CudaBackend (even if CUDA is unavailable on the
        # system).  As many of our types/providers depend on the CUDA
        # runtime we import these here, locally, at the time of
        # instantiation.
        from pyfr.backends.cuda import (blasext, cublas, packing, pointwise,
                                        queue, types)

        # Register our data types
        self.const_matrix_cls = types.CudaConstMatrix
        self.matrix_cls = types.CudaMatrix
        self.matrix_bank_cls = types.CudaMatrixBank
        self.matrix_rslice_cls = types.CudaMatrixRSlice
        self.mpi_matrix_cls = types.CudaMPIMatrix
        self.mpi_view_cls = types.CudaMPIView
        self.queue_cls = queue.CudaQueue
        self.view_cls = types.CudaView

        # Instantiate the kernel providers
        kprovs = [blockmats.BlockDiagMatrixKernels,
                  pointwise.CudaPointwiseKernels,
                  blasext.CudaBlasExtKernels,
                  packing.CudaPackingKernels,
                  cublas.CudaCublasKernels]
        self._providers = [k(self, cfg) for k in kprovs]

        # Numeric data type
        prec = cfg.get('backend', 'precision', 'double')
        if prec not in {'single', 'double'}:
            raise ValueError('CUDA backend precision must be either single or '
                             'double')

        # Convert to a numpy data type
        self.fpdtype = np.dtype(prec).type
