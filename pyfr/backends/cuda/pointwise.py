# -*- coding: utf-8 -*-

import numpy as np

from pycuda.compiler import SourceModule

from pyfr.backends.cuda.provider import CudaKernelProvider
from pyfr.backends.cuda.queue import CudaComputeKernel
from pyfr.util import npdtype_to_ctype

class CudaPointwiseKernels(CudaKernelProvider):
    def __init__(self, backend):
        pass

    def _modopts(self, dtype):
        return dict(dtype=npdtype_to_ctype(dtype))

    def tdisf_inv_3d(self, u, smats, f, gamma):
        nupts, neles = u.nrow, u.ncol / 5

        fn = self._get_function('pointwise', 'tdisf_inv_3d', [np.int32]*2 +
                                [np.intp]*3 + [u.dtype] + [np.int32]*3,
                                self._modopts(u.dtype))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, neles)

        class TdisInv3dKernel(CudaComputeKernel):
            def __call__(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, nupts, neles,
                                       u.data, smats.data, f.data, gamma,
                                       u.leaddim, smats.leaddim, f.leaddim)

        return TdisInv3dKernel()

    def rsolve_rus_inv_3d_int(self, ul_v, pnl_v, ur_v, pnr_v, gamma):
        ninters = ul_v.ncol
        dtype = ul_v.refdtype

        fn = self._get_function('pointwise', 'rsolve_rus_inv_3d_int',
                                [np.int32] + [np.intp]*8 + [dtype],
                                self._modopts(dtype))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, ninters)

        class RsolveRusInv3dIntKernel(CudaComputeKernel):
            def __call__(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, ninters,
                                       ul_v.mapping.data, ul_v.strides.data,
                                       ur_v.mapping.data, ur_v.strides.data,
                                       pnl_v.mapping.data, pnr_v.strides.data,
                                       pnr_v.mapping.data, pnr_v.strides.data,
                                       gamma)

        return RsolveRusInv3dIntKernel()

    def rsolve_rus_inv_3d_mpi(self, ul_mpiv, ur_mpim, pnl_v, gamma):
        ninters = ul_mpiv.ncol
        ul_v = ul_mpiv.view
        dtype = ul_v.refdtype

        fn = self._get_function('pointwise', 'rsolve_rus_inv_3d_mpi',
                                [np.int32] + [np.intp]*5 + [dtype],
                                self._modopts(dtype))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, ninters)

        class RsolveRusInv3dMPIKernel(CudaComputeKernel):
            def __call__(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, ninters,
                                       ul_v.mapping.data, ul_v.strides.data,
                                       ur_mpim.data,
                                       pnl_v.mapping.data, pnl_v.strides.data,
                                       gamma)

        return RsolveRusInv3dMPIKernel()

    def divconf_3d(self, dv, rcpdjac):
        nupts, neles = dv.nrow, dv.ncol / 5

        fn = self._get_function('pointwise', 'divconf_3d', 'iiPPii',
                                self._modopts(dv.dtype))

        grid, block = self._get_2d_grid_block(fn, nupts, neles)

        class Divconf3d(CudaComputeKernel):
            def __call__(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, nupts, neles,
                                       dv.data, rcpdjac.data,
                                       dv.leaddim, rcpdjac.leaddim)

        return Divconf3d()
