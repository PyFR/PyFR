# -*- coding: utf-8 -*-

import numpy as np

import pycuda.driver as cuda

from pyfr.backends.cuda.provider import CudaKernelProvider
from pyfr.backends.cuda.queue import CudaComputeKernel
from pyfr.util import npdtype_to_ctype

class CudaPointwiseKernels(CudaKernelProvider):
    def __init__(self, backend):
        pass

    def _modopts(self, dtype, ndims, nvars):
        return dict(dtype=npdtype_to_ctype(dtype), ndims=ndims, nvars=nvars)

    def tdisf_inv(self, ndims, nvars, u, smats, f, gamma):
        nupts, neles = u.nrow, u.ncol / nvars

        fn = self._get_function('tflux_inv', 'tdisf_inv', [np.int32]*2 +
                                [np.intp]*3 + [u.dtype] + [np.int32]*3,
                                self._modopts(u.dtype, ndims, nvars))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, neles)

        class TdisInvKernel(CudaComputeKernel):
            def run(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, nupts, neles,
                                       u.data, smats.data, f.data, gamma,
                                       u.leaddim, smats.leaddim, f.leaddim)

        return TdisInvKernel()

    def tdisf_vis(self, ndims, nvars, u, smats, rcpdjac, tgradu,
                  gamma, mu, pr):
        nupts, neles = u.nrow, u.ncol / nvars
        rcppr = 1.0 / pr

        fn = self._get_function('tflux_vis', 'tdisf_vis', [np.int32]*2 +
                                [np.intp]*4 + [u.dtype]*3 + [np.int32]*4,
        self._modopts(u.dtype, ndims, nvars))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, neles)

        class TFluxVisKernel(CudaComputeKernel):
            def run(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, nupts, neles,
                                       u.data, smats.data,  rcpdjac.data,
                                       tgradu.data, gamma, mu, rcppr,
                                       u.leaddim, smats.leaddim,
                                       rcpdjac.leaddim, tgradu.leaddim)

        return TFluxVisKernel()

    def conu_int(self, ndims, nvars, ul_vin, ur_vin, ul_vout, ur_vout, beta):
        ninters = ul_vin.ncol
        dtype = ul_vin.refdtype

        fn = self._get_function('coru', 'coru_int', [np.int32] +
                                [np.intp]*6 + [dtype],
                                self._modopts(dtype, ndims, nvars))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, ninters)

        class ConUIntKernel(CudaComputeKernel):
            def run(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, ninters,
                                       ul_vin.mapping.data, ul_vin.strides.data,
                                       ur_vin.mapping.data, ur_vin.strides.data,
                                       ul_vout.mapping.data,
                                       ur_vout.mapping.data, beta)

        return ConUIntKernel()

    def conu_mpi(self, ndims, nvars, ul_vin, ul_vout, ur_mpim, beta):
        ninters = ul_vin.ncol
        dtype = ul_vin.refdtype

        fn = self._get_function('coru', 'coru_mpi', [np.int32] +
                                [np.intp]*4 + [dtype],
                                self._modopts(dtype, ndims, nvars))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, ninters)

        class ConUMPIKernel(CudaComputeKernel):
            def run(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, ninters,
                                       ul_vin.mapping.data, ul_vin.strides.data,
                                       ul_vout.mapping.data, ur_mpim.data,
                                       beta)

        return ConUMPIKernel()

    def gradcoru(self, ndims, nvars, jmats, gradu):
        nfpts, neles = jmats.nrow, gradu.ncol / nvars

        fn = self._get_function('gradcoru', 'gradcoru', 'iiPPii',
                                self._modopts(u.dtype, ndims, nvars))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, neles)

        class GradCorUKernel(CudaComputeKernel):
            def run(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, nfpts, neles,
                                       jmats.data, gradu.data,
                                       jmats.leaddim, gradu.leaddim)

        return GradCorUKernel()

    def rsolve_rus_inv_int(self, ndims, nvars, ul_v, ur_v, magl, magr,
                           normpnorml, gamma):
        ninters = ul_v.ncol
        dtype = ul_v.refdtype

        fn = self._get_function('rsolve_inv', 'rsolve_rus_inv_int',
                                [np.int32] + [np.intp]*7 + [dtype],
                                self._modopts(dtype, ndims, nvars))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, ninters)

        class RsolveRusInvIntKernel(CudaComputeKernel):
            def run(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, ninters,
                                       ul_v.mapping.data, ul_v.strides.data,
                                       ur_v.mapping.data, ur_v.strides.data,
                                       magl.data, magr.data,
                                       normpnorml.data, gamma)

        return RsolveRusInvIntKernel()

    def rsolve_rus_inv_mpi(self, ndims, nvars, ul_mpiv, ur_mpim, magl,
                           normpnorml, gamma):
        ninters = ul_mpiv.ncol
        ul_v = ul_mpiv.view
        dtype = ul_v.refdtype

        fn = self._get_function('rsolve_inv', 'rsolve_rus_inv_mpi',
                                [np.int32] + [np.intp]*5 + [dtype],
                                self._modopts(dtype, ndims, nvars))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, ninters)

        class RsolveRusInvMPIKernel(CudaComputeKernel):
            def run(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, ninters,
                                       ul_v.mapping.data, ul_v.strides.data,
                                       ur_mpim.data,
                                       magl.data, normpnorml.data, gamma)

        return RsolveRusInvMPIKernel()

    def negdivconf(self, ndims, nvars, dv, rcpdjac):
        nupts, neles = dv.nrow, dv.ncol / nvars

        fn = self._get_function('negdivconf', 'negdivconf', 'iiPPii',
                                self._modopts(dv.dtype, ndims, nvars))

        block = (256, 1, 1)
        grid = self._get_grid_for_block(block, neles)

        class Negdivconf(CudaComputeKernel):
            def run(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, nupts, neles,
                                       dv.data, rcpdjac.data,
                                       dv.leaddim, rcpdjac.leaddim)

        return Negdivconf()
