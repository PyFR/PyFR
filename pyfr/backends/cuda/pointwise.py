# -*- coding: utf-8 -*-

import numpy as np

import pycuda.driver as cuda

from pyfr.backends.cuda.provider import CUDAKernelProvider, get_grid_for_block
from pyfr.nputil import npdtype_to_ctype


class CUDAPointwiseKernels(CUDAKernelProvider):
    def _get_function(self, mod, func, argt, opts, nvccopts=None):
        basefn = super(CUDAPointwiseKernels, self)._get_function

        # Map dtype
        nopts = opts.copy()
        nopts['dtype'] = npdtype_to_ctype(nopts['dtype'])

        return basefn(mod, func, argt, nopts, nvccopts)

    def tdisf_inv(self, ndims, nvars, u, smats, f, c):
        nupts, neles = u.nrow, u.ncol / nvars
        opts = dict(dtype=u.dtype, ndims=ndims, nvars=nvars, c=c)

        fn = self._get_function('flux_inv', 'tdisf_inv', 'iiPPPiii', opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, neles)

        return self._basic_kernel(fn, grid, block, nupts, neles,
                                  u, smats, f, u.leaddim,
                                  smats.leaddim, f.leaddim)

    def tdisf_vis(self, ndims, nvars, u, smats, rcpdjac, tgradu, c):
        nupts, neles = u.nrow, u.ncol / nvars
        opts = dict(dtype=u.dtype, ndims=ndims, nvars=nvars, c=c)

        fn = self._get_function('flux_vis', 'tdisf_vis', 'iiPPPPiiii', opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, neles)

        return self._basic_kernel(fn, grid, block, nupts, neles,
                                  u, smats, rcpdjac, tgradu,
                                  u.leaddim, smats.leaddim,
                                  rcpdjac.leaddim, tgradu.leaddim)

    def conu_int(self, nvars, ul_vin, ur_vin, ul_vout, ur_vout, c):
        ninters = ul_vin.ncol
        dtype = ul_vin.refdtype
        opts = dict(dtype=dtype, nvars=nvars, c=c)

        fn = self._get_function('conu', 'conu_int', 'iPPPPPP', opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, ninters)

        return self._basic_kernel(fn, grid, block, ninters,
                                  ul_vin.mapping, ul_vin.strides,
                                  ur_vin.mapping, ur_vin.strides,
                                  ul_vout.mapping, ur_vout.mapping)

    def conu_mpi(self, nvars, ul_vin, ur_mpim, ul_vout, c):
        ninters = ul_vin.ncol
        dtype = ul_vin.view.refdtype
        opts = dict(dtype=dtype, nvars=nvars, c=c)

        fn = self._get_function('conu', 'conu_mpi', 'iPPPP', opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, ninters)

        return self._basic_kernel(fn, grid, block, ninters,
                                  ul_vin.view.mapping, ul_vin.view.strides,
                                  ul_vout.mapping, ur_mpim)

    def conu_bc(self, ndims, nvars, bctype, ul_vin, ul_vout, c):
        ninters = ul_vin.ncol
        dtype = ul_vin.refdtype
        opts = dict(dtype=dtype, ndims=ndims, nvars=nvars, c=c, bctype=bctype)

        fn = self._get_function('conu', 'conu_bc', 'iPPP', opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, ninters)

        return self._basic_kernel(fn, grid, block, ninters,
                                  ul_vin.mapping, ul_vin.strides,
                                  ul_vout.mapping)

    def gradcoru(self, ndims, nvars, jmats, gradu):
        nfpts, neles = jmats.nrow, gradu.ncol / nvars
        opts = dict(dtype=gradu.dtype, ndims=ndims, nvars=nvars)

        fn = self._get_function('gradcoru', 'gradcoru', 'iiPPii', opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, neles)

        return self._basic_kernel(fn, grid, block, nfpts, neles,
                                  jmats, gradu, jmats.leaddim, gradu.leaddim)

    def rsolve_inv_int(self, ndims, nvars, rsinv, ul_v, ur_v, magl, magr,
                       normpnorml, c):
        ninters = ul_v.ncol
        dtype = ul_v.refdtype
        opts = dict(dtype=dtype, ndims=ndims, nvars=nvars, c=c,
                    rsinv=rsinv)

        fn = self._get_function('rsolve_inv', 'rsolve_inv_int', 'iPPPPPPP',
                                opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, ninters)

        return self._basic_kernel(fn, grid, block, ninters,
                                  ul_v.mapping, ul_v.strides,
                                  ur_v.mapping, ur_v.strides,
                                  magl, magr, normpnorml)

    def rsolve_inv_mpi(self, ndims, nvars, rsinv, ul_mpiv, ur_mpim, magl,
                       normpnorml, c):
        ninters = ul_mpiv.ncol
        ul_v = ul_mpiv.view
        dtype = ul_v.refdtype
        opts = dict(dtype=dtype, ndims=ndims, nvars=nvars, c=c, rsinv=rsinv)

        fn = self._get_function('rsolve_inv', 'rsolve_inv_mpi', 'iPPPPP',
                                opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, ninters)

        return self._basic_kernel(fn, grid, block, ninters,
                                  ul_v.mapping, ul_v.strides, ur_mpim,
                                  magl, normpnorml)

    def rsolve_inv_bc(self, ndims, nvars, rsinv, bctype, ul_v, magl,
                      normpnorml, c):
        ninters = ul_v.ncol
        dtype = ul_v.refdtype
        opts = dict(dtype=dtype, ndims=ndims, nvars=nvars, c=c, rsinv=rsinv,
                    bctype=bctype)

        fn = self._get_function('rsolve_inv', 'rsolve_inv_bc', 'iPPPP', opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, ninters)

        return self._basic_kernel(fn, grid, block, ninters,
                                  ul_v.mapping, ul_v.strides,
                                  magl, normpnorml)

    def rsolve_ldg_vis_int(self, ndims, nvars, rsinv, ul_v, gul_v, ur_v, gur_v,
                           magl, magr, normpnorml, c):
        ninters = ul_v.ncol
        opts = dict(dtype=ul_v.refdtype, ndims=ndims, nvars=nvars, c=c,
                    rsinv=rsinv)

        fn = self._get_function('rsolve_vis', 'rsolve_ldg_vis_int',
                                'iPPPPPPPPPPP', opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, ninters)

        return self._basic_kernel(fn, grid, block, ninters,
                                  ul_v.mapping, ul_v.strides,
                                  gul_v.mapping, gul_v.strides,
                                  ur_v.mapping, ur_v.strides,
                                  gur_v.mapping, gur_v.strides,
                                  magl, magr, normpnorml)

    def rsolve_ldg_vis_mpi(self, ndims, nvars, rsinv, ul_mpiv, gul_mpiv,
                           ur_mpim, gur_mpim, magl, normpnorml, c):
        ninters = ul_mpiv.ncol
        ul_v, gul_v = ul_mpiv.view, gul_mpiv.view
        opts = dict(dtype=ul_v.refdtype, ndims=ndims, nvars=nvars, c=c,
                    rsinv=rsinv)

        fn = self._get_function('rsolve_vis', 'rsolve_ldg_vis_mpi',
                                'iPPPPPPPP', opts)
        fn.set_cache_config(cuda.func_cache.PREFER_L1)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, ninters)

        return self._basic_kernel(fn, grid, block, ninters,
                                  ul_v.mapping, ul_v.strides,
                                  gul_v.mapping, gul_v.strides,
                                  ur_mpim, gur_mpim, magl, normpnorml)

    def rsolve_ldg_vis_bc(self, ndims, nvars, rsinv, bctype, ul_v, gul_v, magl,
                          normpnorml, c):
        ninters = ul_v.ncol
        opts = dict(dtype=ul_v.refdtype, ndims=ndims, nvars=nvars, c=c,
                    rsinv=rsinv, bctype=bctype)

        fn = self._get_function('rsolve_vis', 'rsolve_ldg_vis_bc', 'iPPPPPP',
                                opts)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, ninters)

        return self._basic_kernel(fn, grid, block, ninters,
                                  ul_v.mapping, ul_v.strides,
                                  gul_v.mapping, gul_v.strides,
                                  magl, normpnorml)

    def negdivconf(self, nvars, dv, rcpdjac):
        nupts, neles = dv.nrow, dv.ncol / nvars
        opts = dict(dtype=dv.dtype, nvars=nvars)

        fn = self._get_function('negdivconf', 'negdivconf', 'iiPPii', opts)

        block = (256, 1, 1)
        grid = get_grid_for_block(block, neles)

        return self._basic_kernel(fn, grid, block, nupts, neles, dv, rcpdjac,
                                  dv.leaddim, rcpdjac.leaddim)
