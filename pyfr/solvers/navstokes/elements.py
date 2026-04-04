import numpy as np

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    # Use the density field for shock sensing
    shockvar = 'rho'

    @staticmethod
    def grad_con_to_pri(cons, grad_cons, cfg):
        rho, *rhouvw = cons[:-1]
        grad_rho, *grad_rhouvw, grad_E = grad_cons

        # Divide momentum components by ρ
        uvw = [rhov / rho for rhov in rhouvw]

        # Velocity gradients: ∇u⃗ = 1/ρ·[∇(ρu⃗) - u⃗ ⊗ ∇ρ]
        grad_uvw = [(grad_rhov - v*grad_rho) / rho
                    for grad_rhov, v in zip(grad_rhouvw, uvw)]

        # Pressure gradient: ∇p = (γ - 1)·[∇E - 1/2*(u⃗·∇(ρu⃗) - ρu⃗·∇u⃗)]
        gamma = cfg.getfloat('constants', 'gamma')
        grad_p = grad_E - 0.5*(np.einsum('ijk,iljk->ljk', uvw, grad_rhouvw) +
                               np.einsum('ijk,iljk->ljk', rhouvw, grad_uvw))
        grad_p *= (gamma - 1)

        return [grad_rho, *grad_uvw, grad_p]

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Can elide interior flux calculations at p = 0
        if self.basis.order == 0:
            return

        # Register our flux kernels
        kprefix = 'pyfr.solvers.navstokes.kernels'
        self._be.pointwise.register(f'{kprefix}.tflux')

        # Handle shock capturing and Sutherland's law
        shock_capturing = self.cfg.get('solver', 'shock-capturing')
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
            'interp_expr': self.basis.interp_expr,
            'shock_capturing': shock_capturing,
            'visc_corr': visc_corr
        }

        # Helpers
        r, s = self.mesh_regions, self._slice_mat

        # Mode-dependent setup
        if self.grad_fusion:
            pts, fused = 'upts', True
            kname = 'tdisf_fused'
        elif 'flux' in self.antialias:
            pts, fused = 'qpts', False
            kname = 'tdisf'
        else:
            pts, fused = 'upts', False
            kname = 'tdisf'

        npts = self.nqpts if pts == 'qpts' else self.nupts
        has_uin = pts == 'upts'

        # Build per-region kernel args
        tdisf = []
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

            # Reference point coordinates (for linear smats and AV interp)
            kw['upts'] = getattr(self, pts)

            if has_uin:
                kw['f'] = s(self._vect_upts, rgn)
                if fused:
                    kw['gradu'] = s(self._grad_upts, rgn)
            else:
                kw['u'] = s(self._scal_qpts, rgn)
                kw['f'] = s(self._vect_qpts, rgn)

            tdisf.append((ktype, r[rgn], rgn, kw))

        if has_uin:
            def tdisf_k(uin):
                return self._make_sliced_kernel(
                    self._be.kernel(
                        'tflux', tplargs=tplargs | {'ktype': kt},
                        dims=[npts, n], u=s(self.scal_upts[uin], rgn),
                        artvisc_vtx=self.vtx_views.get(rgn), **kw
                    )
                    for kt, n, rgn, kw in tdisf
                )
        else:
            def tdisf_k():
                return self._make_sliced_kernel(
                    self._be.kernel(
                        'tflux', tplargs=tplargs | {'ktype': kt},
                        dims=[npts, n],
                        artvisc_vtx=self.vtx_views.get(rgn), **kw
                    )
                    for kt, n, rgn, kw in tdisf
                )

        self.kernels[kname] = tdisf_k
