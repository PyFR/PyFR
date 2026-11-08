from pyfr.exprs import resolve_aux
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    # Use the density field for shock sensing
    shockvar = 'rho'

    @classmethod
    def stats_tables(cls, cfg):
        return super().stats_tables(cfg) + ['pyfr.solvers.navstokes']

    @classmethod
    def auxvars(cls, ndims, cfg):
        aux = super().auxvars(ndims, cfg)
        c = cfg.items_as('constants', float)

        if cfg.get('solver', 'viscosity-correction', 'none') == 'sutherland':
            m, tr, ts = c['mu'], c['cpTref'], c['cpTs']
            aux['mu'] = f'{m*(tr + ts)}*pow(cpT/{tr}, 1.5)/(cpT + {ts})'
        else:
            aux['mu'] = str(c['mu'])

        aux['nu'] = 'mu/rho'

        return resolve_aux(aux)

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Can elide interior flux calculations at p = 0
        if self.basis.order == 0:
            return

        # Register our flux kernels
        kprefix = 'pyfr.solvers.navstokes.kernels'
        self._be.pointwise.register(f'{kprefix}.tflux')

        # Handle Sutherland's law
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        if visc_corr not in {'sutherland', 'none'}:
            raise ValueError('Invalid viscosity-correction option')

        # Template parameters for the flux kernels
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs,
            'visc_corr': visc_corr
        }

        # Helpers
        r, s = self.mesh_regions, self._slice_mat

        # Mode-dependent setup
        if self.grad_fusion:
            pts, fused = 'upts', True
            kname = 'disf_fused'
        elif self.fluxaa:
            pts, fused = 'qpts', False
            kname = 'disf'
        else:
            pts, fused = 'upts', False
            kname = 'disf'

        npts = self.nqpts if pts == 'qpts' else self.nupts
        has_uin = pts == 'upts'

        # Build per-region kernel args
        disf = []
        for rgn in ('curved', 'linear'):
            if rgn not in r:
                continue

            ktype = f'{rgn}-fused' if fused else rgn

            # Region-specific geometry kwargs
            kw = {}
            if rgn == 'curved':
                kw['smats'] = self.curved_smat_at(pts)
                if fused:
                    kw['rcpdjac'] = self.rcpdjac_at('upts', 'curved')
            else:
                kw['verts'] = self.ploc_at('linspts', 'linear')

            # Reference point coordinates for the linear smats
            kw['upts'] = getattr(self, pts)

            if has_uin:
                kw['f'] = s(self._vect_upts, rgn)
                if fused:
                    kw['gradu'] = s(self._grad_upts, rgn)
            else:
                kw['u'] = s(self._scal_qpts, rgn)
                kw['f'] = s(self._vect_qpts, rgn)

            disf.append((ktype, r[rgn], rgn, kw))

        if has_uin:
            def disf_k(uin):
                sc = self.shock_capturing
                return self._make_sliced_kernel(
                    self._be.kernel(
                        'tflux', tplargs=tplargs | sc.tplargs | {'ktype': kt},
                        dims=[npts, n], u=s(self.scal_upts[uin], rgn),
                        **kw, **sc.flux_args(self, rgn, pts)
                    )
                    for kt, n, rgn, kw in disf
                )
        else:
            def disf_k():
                sc = self.shock_capturing
                return self._make_sliced_kernel(
                    self._be.kernel(
                        'tflux', tplargs=tplargs | sc.tplargs | {'ktype': kt},
                        dims=[npts, n], **kw, **sc.flux_args(self, rgn, pts)
                    )
                    for kt, n, rgn, kw in disf
                )

        self.kernels[kname] = disf_k
