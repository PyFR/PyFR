# -*- coding: utf-8 -*-

from pyfr.exc import PyFRInvalidKernelError

from pyfr.backends.cuda.types import (CudaMatrix, CudaConstMatrix,
                                      CudaSparseMatrix, CudaView, CudaHostView)

from pyfr.backends.cuda.packing import CudaPackingKernels
from pyfr.backends.cuda.cublas import CudaCublasKernels
from pyfr.backends.cuda.sample import CudaSampleKernels

from pyfr.backends.cuda.queue import CudaQueue

class CudaBackend(object):
    def __init__(self):
        self._providers = [kprov(self) for kprov in [CudaSampleKernels,
                                                     CudaPackingKernels,
                                                     CudaCublasKernels]]


    def matrix(self, *args, **kwargs):
        return CudaMatrix(self, *args, **kwargs)

    def const_matrix(self, *args, **kwargs):
        return CudaConstMatrix(self, *args, **kwargs)

    def sparse_matrix(self, *args, **kwargs):
        return CudaSparseMatrix(self, *args, **kwargs)

    def view(self, *args, **kwargs):
        return CudaView(self, *args, **kwargs)

    def host_view(self, *args, **kwargs):
        return CudaHostView(self, *args, **kwargs)

    def queue(self):
        return CudaQueue()

    def kernel(self, kname, *args, **kwargs):
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
