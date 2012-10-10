# -*- coding: utf-8 -*-

from pycuda.compiler import SourceModule

from pyfr.backends.cuda.provider import CudaKernelProvider
from pyfr.backends.cuda.queue import CudaComputeKernel

from pyfr.util import npdtype_to_ctype

class CudaSampleKernels(CudaKernelProvider):
    def __init__(self, backend):
        pass

    def square(self, mat):
        fn = self._get_function('square', 'square', 'Piii',
                                dict(mat_order=mat.order,
                                     mat_ctype=npdtype_to_ctype(mat.dtype)))

        grid, block = self._get_2d_grid_block(fn, mat.nrow, mat.ncol)

        class SquareKernel(CudaComputeKernel):
            def __call__(self, stream):
                fn.prepared_async_call(grid, block, stream, mat.data,
                                       mat.nrow, mat.ncol, mat.leaddim)

        return SquareKernel()
