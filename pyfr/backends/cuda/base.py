# -*- coding: utf-8 -*-

from pyfr.exc import PyFRInvalidKernelError

from pyfr.backends.base import Backend

from pyfr.backends.cuda.types import (CudaMatrix, CudaMatrixBank,
                                      CudaConstMatrix, CudaSparseMatrix,
                                      CudaView, CudaMPIMatrix, CudaMPIView)

from pyfr.backends.cuda.packing import CudaPackingKernels
from pyfr.backends.cuda.cublas import CudaCublasKernels
from pyfr.backends.cuda.sample import CudaSampleKernels

from pyfr.backends.cuda.queue import CudaQueue

class CudaBackend(Backend):
    def __init__(self):
        super(CudaBackend, self).__init__()
        self._providers = [kprov(self) for kprov in [CudaSampleKernels,
                                                     CudaPackingKernels,
                                                     CudaCublasKernels]]


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
                return getattr(prov, kname)(*args, **kwargs)
            except (AttributeError, TypeError, ValueError):
                pass
        else:
            raise PyFRInvalidKernelError("'{}' has no providers"\
                                         .format(kname))

    def runall(self, queues):
        CudaQueue.runall(queues)
