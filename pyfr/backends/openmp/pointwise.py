# -*- coding: utf-8 -*-

from pyfr.backends.base import traits
from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.nputil import npdtype_to_ctype


class OpenMPPointwiseKernels(OpenMPKernelProvider):
    def _get_function(self, mod, func, rest, argt, opts):
        basefn = super(OpenMPPointwiseKernels, self)._get_function

        # Map dtype
        nopts = opts.copy()
        nopts['dtype'] = npdtype_to_ctype(nopts['dtype'])

        return basefn(mod, func, rest, argt, nopts)

    @traits(u={'align'}, smats={'align'}, f={'align'})
    def tdisf_inv(self, ndims, nvars, u, smats, f, c):
        nupts, neles = u.nrow, u.soa_shape[2]
        opts = dict(dtype=u.dtype, ndims=ndims, nvars=nvars, c=c)

        fn = self._get_function('flux_inv', 'tdisf_inv', None, 'iiPPPiii',
                                opts)

        return self._basic_kernel(fn, nupts, neles, u, smats, f,
                                  u.leadsubdim, smats.leadsubdim, f.leadsubdim)

    @traits(u={'align'}, smats={'align'}, rcpdjac={'align'}, tgradu={'align'})
    def tdisf_vis(self, ndims, nvars, u, smats, rcpdjac, tgradu, c):
        nupts, neles = u.nrow, u.soa_shape[2]
        opts = dict(dtype=u.dtype, ndims=ndims, nvars=nvars, c=c)

        fn = self._get_function('flux_vis', 'tdisf_vis', None,
                                'iiPPPPiiii', opts)

        return self._basic_kernel(fn, nupts, neles, u, smats, rcpdjac, tgradu,
                                  rcpdjac.leaddim, u.leadsubdim,
                                  smats.leadsubdim, tgradu.leadsubdim)

    def conu_int(self, nvars, ul_vin, ur_vin, ul_vout, ur_vout, c):
        ninters = ul_vin.ncol
        dtype = ul_vin.refdtype
        opts = dict(dtype=dtype, nvars=nvars, c=c)

        fn = self._get_function('conu', 'conu_int', None, 'iPPPPPP', opts)

        return self._basic_kernel(fn, ninters,
                                  ul_vin.mapping, ul_vin.strides,
                                  ur_vin.mapping, ur_vin.strides,
                                  ul_vout.mapping, ur_vout.mapping)

    def conu_mpi(self, nvars, ul_vin, ur_mpim, ul_vout, c):
        ninters = ul_vin.ncol
        dtype = ul_vin.view.refdtype
        opts = dict(dtype=dtype, nvars=nvars, c=c)

        fn = self._get_function('conu', 'conu_mpi', None, 'iPPPP', opts)

        return self._basic_kernel(fn, ninters,
                                  ul_vin.view.mapping, ul_vin.view.strides,
                                  ul_vout.mapping, ur_mpim)

    def conu_bc(self, ndims, nvars, bctype, ul_vin, ul_vout, c):
        ninters = ul_vin.ncol
        dtype = ul_vin.refdtype
        opts = dict(dtype=dtype, ndims=ndims, nvars=nvars, c=c, bctype=bctype)

        fn = self._get_function('conu', 'conu_bc', None, 'iPPP', opts)

        return self._basic_kernel(fn, ninters,
                                  ul_vin.mapping, ul_vin.strides,
                                  ul_vout.mapping)

    @traits(jmats={'align'}, gradu={'align'})
    def gradcoru(self, ndims, nvars, jmats, gradu):
        nfpts, neles = jmats.nrow, gradu.ncol / nvars
        opts = dict(dtype=gradu.dtype, ndims=ndims, nvars=nvars)

        fn = self._get_function('gradcoru', 'gradcoru', None, 'iiPPii', opts)

        return self._basic_kernel(fn, nfpts, neles, jmats, gradu,
                                  jmats.leadsubdim, gradu.leadsubdim)

    def rsolve_inv_int(self, ndims, nvars, rsinv, ul_v, ur_v, magl, magr,
                       normpnorml, c):
        ninters = ul_v.ncol
        dtype = ul_v.refdtype
        opts = dict(dtype=dtype, ndims=ndims, nvars=nvars, c=c,
                    rsinv=rsinv)

        fn = self._get_function('rsolve_inv', 'rsolve_inv_int', None,
                                'iPPPPPPP', opts)

        return self._basic_kernel(fn, ninters, ul_v.mapping, ul_v.strides,
                                  ur_v.mapping, ur_v.strides, magl, magr,
                                  normpnorml)

    def rsolve_inv_mpi(self, ndims, nvars, rsinv, ul_mpiv, ur_mpim, magl,
                       normpnorml, c):
        ninters = ul_mpiv.ncol
        ul_v = ul_mpiv.view
        dtype = ul_v.refdtype
        opts = dict(dtype=dtype, ndims=ndims, nvars=nvars, c=c, rsinv=rsinv)

        fn = self._get_function('rsolve_inv', 'rsolve_inv_mpi', None,
                                'iPPPPP', opts)

        return self._basic_kernel(fn, ninters, ul_v.mapping, ul_v.strides,
                                  ur_mpim, magl, normpnorml)

    def rsolve_inv_bc(self, ndims, nvars, rsinv, bctype, ul_v, magl,
                      normpnorml, c):
        ninters = ul_v.ncol
        dtype = ul_v.refdtype
        opts = dict(dtype=dtype, ndims=ndims, nvars=nvars, c=c, rsinv=rsinv,
                    bctype=bctype)

        fn = self._get_function('rsolve_inv', 'rsolve_inv_bc', None,
                                'iPPPP', opts)

        return self._basic_kernel(fn, ninters,
                                  ul_v.mapping, ul_v.strides,
                                  magl, normpnorml)

    def rsolve_ldg_vis_int(self, ndims, nvars, rsinv, ul_v, gul_v, ur_v, gur_v,
                           magl, magr, normpnorml, c):
        ninters = ul_v.ncol
        opts = dict(dtype=ul_v.refdtype, ndims=ndims, nvars=nvars, c=c,
                    rsinv=rsinv)

        fn = self._get_function('rsolve_vis', 'rsolve_ldg_vis_int', None,
                                'iPPPPPPPPPPP', opts)

        return self._basic_kernel(fn, ninters,
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

        fn = self._get_function('rsolve_vis', 'rsolve_ldg_vis_mpi', None,
                                'iPPPPPPPP', opts)

        return self._basic_kernel(fn, ninters,
                                  ul_v.mapping, ul_v.strides,
                                  gul_v.mapping, gul_v.strides,
                                  ur_mpim, gur_mpim, magl, normpnorml)

    def rsolve_ldg_vis_bc(self, ndims, nvars, rsinv, bctype, ul_v, gul_v, magl,
                          normpnorml, c):
        ninters = ul_v.ncol
        opts = dict(dtype=ul_v.refdtype, ndims=ndims, nvars=nvars, c=c,
                    rsinv=rsinv, bctype=bctype)

        fn = self._get_function('rsolve_vis', 'rsolve_ldg_vis_bc', None,
                                'iPPPPPP', opts)

        return self._basic_kernel(fn, ninters,
                                  ul_v.mapping, ul_v.strides,
                                  gul_v.mapping, gul_v.strides,
                                  magl, normpnorml)

    @traits(dv={'align'}, rcpdjac={'align'})
    def negdivconf(self, nvars, dv, rcpdjac):
        nupts, neles = dv.nrow, dv.soa_shape[2]
        opts = dict(dtype=dv.dtype, nvars=nvars)

        fn = self._get_function('negdivconf', 'negdivconf', None, 'iiPPii',
                                opts)


        return self._basic_kernel(fn, nupts, neles, dv, rcpdjac,
                                  rcpdjac.leaddim, dv.leadsubdim)
