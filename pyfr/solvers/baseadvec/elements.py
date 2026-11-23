from pyfr.solvers.base import BaseElements
from pyfr.solvers.baseadvec.operators import BaseAdvectionOperators
from pyfr.solvers.baseadvec.shockcapturing import ShockCapturingMixin


class BaseAdvectionElements(ShockCapturingMixin, BaseElements):
    operatorscls = BaseAdvectionOperators

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
        if self.fluxaa:
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

    def set_external(self, name, spec, value=None):
        self._external_args[name] = spec

        if value is not None:
            self._external_vals[name] = value

    def set_backend(self, backend, nonce):
        super().set_backend(backend, nonce)

        kernels = self.kernels

        # Register pointwise kernels with the backend
        self._be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.negdivconf'
        )

        self._be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.evalsrcmacros'
        )

        # Interpolation from elemental points
        kernels['disu'] = lambda uin: self._be.kernel(
            'mul', self.opmat('M0'), self.scal_upts[uin],
            out=self._scal_fpts
        )

        if self.fluxaa and self.shape.order > 0:
            kernels['qptsu'] = lambda uin: self._be.kernel(
                'mul', self.opmat('M7'), self.scal_upts[uin],
                out=self._scal_qpts
            )

        # First flux correction kernel
        if self.shape.order > 0:
            kernels['divpcorf'] = lambda uin, fout: self.divpcorf(
                self._vect_qpts if self.fluxaa else self._vect_upts,
                self._div_upts(uin, fout)
            )

        # Second flux correction kernel
        kernels['divconf'] = lambda uin, fout: self.divconf(
            self._scal_fpts, self._div_upts(uin, fout),
            add=self.shape.order > 0
        )

        # Transformed to physical divergence kernel + source term
        kernels['negdivconf'] = self._negdivconf

        kernels['evalsrcmacros'] = lambda uin: self._be.kernel(
            'evalsrcmacros', tplargs=self._srctplargs,
            dims=[self.nupts, self.neles], extrns=self._external_args,
            ploc=self.ploc_at('upts') if self._ploc_in_src_macros else None,
            u=self.scal_upts[uin],
            **self._external_vals
        )

    def _preserve_soln(self, uin, fout):
        # Source terms need the solution after the divergence is formed
        return uin == fout and self._soln_in_src_macros

    def _div_upts(self, uin, fout):
        # Keep the solution intact if the divergence would overwrite it
        if self._preserve_soln(uin, fout):
            return self._scal_upts_tmp
        else:
            return self.scal_upts[fout]

    def _negdivconf(self, uin, fout):
        preserve = self._preserve_soln(uin, fout)

        # Kernel type and metric terms which scale the divergence
        ktype, kw = self.div_metrics()
        tplargs = {'preserve_soln': preserve, 'ktype': ktype}

        return self._be.kernel(
            'negdivconf', tplargs=self._srctplargs | tplargs,
            dims=[self.nupts, self.neles], extrns=self._external_args,
            divconf=self._div_upts(uin, fout),
            ploc=self.ploc_at('upts') if self._ploc_in_src_macros else None,
            u=self.scal_upts[uin] if self._soln_in_src_macros else None,
            **kw, **self._external_vals
        )
