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
            'shock_capturing': shock_capturing,
            'visc_corr': visc_corr
        }

        # Helpers
        tdisf = []
        c, l = 'curved', 'linear'
        r, s = self._mesh_regions, self._slice_mat
        av = self.artvisc

        # Gradient + flux kernel fusion
        if self.grad_fusion:
            if c in r:
                tdisf.append(lambda uin: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'curved-fused'},
                    dims=[self.nupts, r[c]], u=s(self.scal_upts[uin], c),
                    artvisc=s(av, c), f=s(self._vect_upts, c),
                    gradu=s(self._grad_upts, c),
                    rcpdjac=self.rcpdjac_at('upts', 'curved'),
                    smats=self.curved_smat_at('upts')
                ))
            if l in r:
                tdisf.append(lambda uin: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'linear-fused'},
                    dims=[self.nupts, r[l]], u=s(self.scal_upts[uin], l),
                    artvisc=s(av, l), f=s(self._vect_upts, l),
                    gradu=s(self._grad_upts, l),
                    verts=self.ploc_at('linspts', l), upts=self.upts
                ))

            def tdisf_k(uin):
                return self._make_sliced_kernel(k(uin) for k in tdisf)

            self.kernels['tdisf_fused'] = tdisf_k
        # No gradient + flux kernel fusion, with flux-AA
        elif 'flux' in self.antialias:
            if c in r:
                tdisf.append(lambda: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'curved'},
                    dims=[self.nqpts, r[c]], u=s(self._scal_qpts, c),
                    f=s(self._vect_qpts, c), artvisc=s(av, c),
                    smats=self.curved_smat_at('qpts')
                ))
            if l in r:
                tdisf.append(lambda: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'linear'},
                    dims=[self.nqpts, r[l]], u=s(self._scal_qpts, l),
                    f=s(self._vect_qpts, l), artvisc=s(av, l),
                    verts=self.ploc_at('linspts', l), upts=self.qpts
                ))

            def tdisf_k():
                return self._make_sliced_kernel(k() for k in tdisf)

            self.kernels['tdisf'] = tdisf_k
        # No gradient + flux kernel fusion, no flux-AA
        else:
            if c in r:
                tdisf.append(lambda uin: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'curved'},
                    dims=[self.nupts, r[c]], u=s(self.scal_upts[uin], c),
                    f=s(self._vect_upts, c), artvisc=s(av, c),
                    smats=self.curved_smat_at('upts')
                ))
            if l in r:
                tdisf.append(lambda uin: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'linear'},
                    dims=[self.nupts, r[l]], u=s(self.scal_upts[uin], l),
                    f=s(self._vect_upts, l), artvisc=s(av, l),
                    verts=self.ploc_at('linspts', l), upts=self.upts
                ))

            def tdisf_k(uin):
                return self._make_sliced_kernel(k(uin) for k in tdisf)

            self.kernels['tdisf'] = tdisf_k
