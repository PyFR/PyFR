from pyfr.backends.base import NullKernel
from pyfr.solvers.base import BaseElements

import numpy as np

class BaseAdvectionElements(BaseElements):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        # Global kernel arguments
        self._external_args = {}
        self._external_vals = {}

        # Source term kernel arguments
        self._srctplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'src_macros': []
        }

        self._ploc_in_src_macros = False
        self._soln_in_src_macros = False

    @property
    def has_src_macros(self):
        return bool(self._srctplargs['src_macros'])

    @property
    def _scratch_bufs(self):
        if 'flux' in self.antialias:
            return {'scal_fpts', 'scal_qpts', 'vect_qpts'}
        else:
            return {'scal_fpts', 'vect_upts'}

    def add_src_macro(self, mod, name, tplargs, ploc=False, soln=False):
        self._ploc_in_src_macros |= ploc
        self._soln_in_src_macros |= soln

        for m, n in self._srctplargs['src_macros']:
            if m == mod or n == name:
                raise RuntimeError(f'Aliased macros in src_macros: {name}')

        for k, v in tplargs.items():
            if k in self._srctplargs and self._srctplargs[k] != v:
                raise RuntimeError(f'Aliased terms in template args: {k}')

        self._srctplargs['src_macros'].append((mod, name))
        self._srctplargs |= tplargs

    def _set_external(self, name, spec, value=None):
        self._external_args[name] = spec

        if value is not None:
            self._external_vals[name] = value

    def set_backend(self, backend, nscalupts, nonce, linoff):
        super().set_backend(backend, nscalupts, nonce, linoff)

        kernels = self.kernels

        # Register pointwise kernels with the backend
        self._be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.negdivconf'
        )

        self._be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.evalsrcmacros'
        )

        # What anti-aliasing options we're running with
        fluxaa = 'flux' in self.antialias

        # Interpolation from elemental points
        kernels['disu'] = lambda uin: self._be.kernel(
            'mul', self.opmat('M0'), self.scal_upts[uin],
            out=self._scal_fpts
        )

        if fluxaa and self.basis.order > 0:
            kernels['qptsu'] = lambda uin: self._be.kernel(
                'mul', self.opmat('M7'), self.scal_upts[uin],
                out=self._scal_qpts
            )

        # First flux correction kernel
        if fluxaa and self.basis.order > 0:
            kernels['tdivtpcorf'] = lambda fout: self._be.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M9'), self._vect_qpts,
                out=self.scal_upts[fout]
            )
        elif self.basis.order > 0:
            kernels['tdivtpcorf'] = lambda fout: self._be.kernel(
                'mul', self.opmat('M1 - M3*M2'), self._vect_upts,
                out=self.scal_upts[fout]
            )

        # Second flux correction kernel
        kernels['tdivtconf'] = lambda fout: self._be.kernel(
            'mul', self.opmat('M3'), self._scal_fpts,
            out=self.scal_upts[fout], beta=float(self.basis.order > 0)
        )

        def copy_soln(uin):
            if self._soln_in_src_macros:
                return self._be.kernel('copy', self._scal_upts_cpy,
                                       self.scal_upts[uin])
            else:
                return NullKernel()

        kernels['copy_soln'] = copy_soln

        # Transformed to physical divergence kernel + source term
        kernels['negdivconf'] = lambda fout: self._be.kernel(
            'negdivconf', tplargs=self._srctplargs,
            dims=[self.nupts, self.neles], extrns=self._external_args,
            tdivtconf=self.scal_upts[fout], rcpdjac=self.rcpdjac_at('upts'),
            ploc=self.ploc_at('upts') if self._ploc_in_src_macros else None,
            u=self._scal_upts_cpy if self._soln_in_src_macros else None,
            **self._external_vals
        )

        kernels['evalsrcmacros'] = lambda uin: self._be.kernel(
            'evalsrcmacros', tplargs=self._srctplargs,
            dims=[self.nupts, self.neles], extrns=self._external_args,
            ploc=self.ploc_at('upts') if self._ploc_in_src_macros else None,
            u=self.scal_upts[uin],
            **self._external_vals
        )

        # In-place solution filter
        if self.cfg.getint('soln-filter', 'nsteps', '0'):
            def modal_filter(uin):
                mul = self._be.kernel(
                    'mul', self.opmat('M10'), self.scal_upts[uin],
                    out=self._scal_upts_temp
                )
                copy = self._be.kernel(
                    'copy', self.scal_upts[uin], self._scal_upts_temp
                )

                return self._be.ordered_meta_kernel([mul, copy])

            kernels['modal_filter'] = modal_filter

        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'entropy-filter':
            tags = {'align'}

            # Allocate one minimum entropy value per interface
            self.nfaces = len(self.nfacefpts)
            ext = nonce + 'entmin_int'

            # Set values to -inf for pre-proc filter to enforce positivity
            entmin_int = np.full((self.nfaces, self.neles),
                                 -self._be.fpdtype_max)
            self.entmin_int = self._be.matrix((self.nfaces, self.neles),
                                              tags=tags, extent=ext,
                                              initval=entmin_int)

            # Setup nodal/modal operator matrices
            form = self.cfg.get('solver-entropy-filter', 'formulation',
                                'nonlinear')
            if form == 'linearised':
                self.invvdm = self.vdm_ef = None
            elif form == 'nonlinear':
                self.invvdm = self._be.const_matrix(self.basis.ubasis.invvdm.T)
                vdmu = self.basis.ubasis.vdm.T
                vdmf = self.basis.ubasis.vdm_at(self.basis.fpts).T
                vdm = vdmu if self.basis.fpts_in_upts else np.vstack((vdmu, vdmf))
                self.vdm_ef = self._be.const_matrix(vdm)
            else:
                raise ValueError('Invalid entropy filter formulation.')

            if self.basis.fpts_in_upts:
                self.m0 = None
            else:
                self.m0 = self._be.const_matrix(self.basis.m0)

    def get_entmin_int_fpts_for_inter(self, eidx, fidx):
        return (self.entmin_int.mid,), (fidx,), (eidx,)

    def get_entmin_bc_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]
        return (self.entmin_int.mid,)*nfp, (fidx,)*nfp, (eidx,)*nfp
