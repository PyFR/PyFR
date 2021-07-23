# -*- coding: utf-8 -*-

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
        uvw = [rhov/rho for rhov in rhouvw]

        # Velocity gradients
        # ∇(\vec{u}) = 1/ρ·[∇(ρ\vec{u}) - \vec{u} \otimes ∇ρ]
        grad_uvw = [(grad_rhov - v*grad_rho)/rho 
                    for grad_rhov, v in zip(grad_rhouvw, uvw)]

        # Pressure gradient
        # ∇p = (gamma - 1)·[∇E - 1/2*(\vec{u}·∇(ρ\vec{u}) - ρ\vec{u}·∇(\vec{u}))]
        gamma = cfg.getfloat('constants', 'gamma')
        grad_p = grad_E - 0.5*(np.einsum('ijk, iljk -> ljk', uvw, grad_rhouvw) +
                               np.einsum('ijk, iljk -> ljk', rhouvw, grad_uvw))
        grad_p *= (gamma - 1)

        return [grad_rho] + grad_uvw + [grad_p]

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Register our flux kernels
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.tflux')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.tfluxlin')

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

        # Common arguments
        if 'flux' in self.antialias:
            u = lambda s: self._slice_mat(self._scal_qpts, s)
            f = lambda s: self._slice_mat(self._vect_qpts, s)
            pts, npts = 'qpts', self.nqpts
        else:
            u = lambda s: self._slice_mat(self.scal_upts_inb, s)
            f = lambda s: self._slice_mat(self._vect_upts, s)
            pts, npts = 'upts', self.nupts

        av = self.artvisc

        # Mesh regions
        regions = self._mesh_regions

        if 'curved' in regions:
            self.kernels['tdisf_curved'] = lambda: self._be.kernel(
                'tflux', tplargs=tplargs, dims=[npts, regions['curved']],
                u=u('curved'), f=f('curved'),
                artvisc=self._slice_mat(av, 'curved') if av else None,
                smats=self.smat_at(pts, 'curved')
            )

        if 'linear' in regions:
            upts = getattr(self, pts)
            self.kernels['tdisf_linear'] = lambda: self._be.kernel(
                'tfluxlin', tplargs=tplargs, dims=[npts, regions['linear']],
                u=u('linear'), f=f('linear'),
                artvisc=self._slice_mat(av, 'linear') if av else None,
                verts=self.ploc_at('linspts', 'linear'), upts=upts
            )
