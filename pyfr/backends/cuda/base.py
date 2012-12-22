# -*- coding: utf-8 -*-

import numpy as np

from pycuda.autoinit import context as cuda_ctx
import pycuda.driver as cuda

from pyfr.exc import PyFRInvalidKernelError

from pyfr.backends.base import Backend

from pyfr.backends.cuda.types import (CudaMatrix, CudaMatrixBank,
                                      CudaConstMatrix, CudaSparseMatrix,
                                      CudaView, CudaMPIMatrix, CudaMPIView)

from pyfr.backends.cuda.packing import CudaPackingKernels
from pyfr.backends.cuda.blasext import CudaBlasExtKernels
from pyfr.backends.cuda.cublas import CudaCublasKernels
from pyfr.backends.cuda.pointwise import CudaPointwiseKernels

from pyfr.backends.cuda.queue import CudaQueue

class CudaBackend(Backend):
    name = 'CUDA'
    packing = 'SoA'

    def __init__(self, cfg):
        super(CudaBackend, self).__init__(cfg)

        # Some CUDA devices share L1 cache and shared memory; on these
        # devices CUDA allows us to specify a preference between L1
        # cache and shared memory.  For the benefit of CUBLAS (which
        # benefits greatly from more shared memory but fails to
        # declare its preference) we set the global default to
        # PREFER_SHARED.
        cuda_ctx.set_cache_config(cuda.func_cache.PREFER_SHARED)

        # Kernel provider classes
        kprovcls = [CudaPointwiseKernels, CudaBlasExtKernels,
                    CudaPackingKernels, CudaCublasKernels]
        self._providers = [k(self) for k in kprovcls]

        # Numeric data type
        prec = cfg.get('backend', 'precision', 'double')
        if prec not in {'single', 'double'}:
            raise ValueError('CUDA backend precision must be either single or'\
                             'double')

        # Convert to a numpy data type
        self.fpdtype = np.dtype(prec).type


    def _matrix(self, *args, **kwargs):
        return CudaMatrix(self, *args, **kwargs)

    def _matrix_bank(self, *args, **kwargs):
        return CudaMatrixBank(self, *args, **kwargs)

    def _const_matrix(self, *args, **kwargs):
        return CudaConstMatrix(self, *args, **kwargs)

    def _sparse_matrix(self, *args, **kwargs):
        return CudaSparseMatrix(self, *args, **kwargs)

    def _is_sparse(self, mat, tags):
        # Currently, no support for sparse matrices
        return False

    def _mpi_matrix(self, *args, **kwargs):
        return CudaMPIMatrix(self, *args, **kwargs)

    def _view(self, *args, **kwargs):
        return CudaView(self, *args, **kwargs)

    def _mpi_view(self, *args, **kwargs):
        return CudaMPIView(self, *args, **kwargs)

    def _queue(self):
        return CudaQueue()

    def _kernel(self, kname, *args, **kwargs):
        for prov in reversed(self._providers):
            try:
                kern = getattr(prov, kname)
            except AttributeError:
                continue

            return kern(*args, **kwargs)
        else:
            raise PyFRInvalidKernelError("'{}' has no providers"\
                                         .format(kname))

    def runall(self, queues):
        CudaQueue.runall(queues)
