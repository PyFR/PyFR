# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend, blockmats


class CUDABackend(BaseBackend):
    name = 'cuda'

    def __init__(self, cfg):
        super(CUDABackend, self).__init__(cfg)

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
        # import the CUDABackend (even if CUDA is unavailable on the
        # system).  As many of our types/providers depend on the CUDA
        # runtime we import these here, locally, at the time of
        # instantiation.
        from pyfr.backends.cuda import (blasext, cublas, packing, pointwise,
                                        types)

        # Register our data types
        self.block_diag_matrix_cls = types.CUDABlockDiagMatrix
        self.const_matrix_cls = types.CUDAConstMatrix
        self.matrix_cls = types.CUDAMatrix
        self.matrix_bank_cls = types.CUDAMatrixBank
        self.matrix_rslice_cls = types.CUDAMatrixRSlice
        self.mpi_matrix_cls = types.CUDAMPIMatrix
        self.mpi_view_cls = types.CUDAMPIView
        self.queue_cls = types.CUDAQueue
        self.view_cls = types.CUDAView

        # Instantiate the kernel providers
        kprovs = [blockmats.BlockDiagMatrixKernels,
                  pointwise.CUDAPointwiseKernels,
                  blasext.CUDABlasExtKernels,
                  packing.CUDAPackingKernels,
                  cublas.CUDACublasKernels]
        self._providers = [k(self, cfg) for k in kprovs]

        # Numeric data type
        prec = cfg.get('backend', 'precision', 'double')
        if prec not in {'single', 'double'}:
            raise ValueError('CUDA backend precision must be either single or '
                             'double')

        # Convert to a numpy data type
        self.fpdtype = np.dtype(prec).type
