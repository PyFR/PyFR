import numpy as np

from pyfr.fluids.base import BaseFluid


class CPGFluid(BaseFluid):
    name = 'cpg'

    @property
    def nvars(self):
        return self.ndims + 2

    def _register_state(self):
        ndims, nvars = self.ndims, self.nvars
        gamma = self.c['gamma']
        reg = self._register_quantity

        reg('rho', (), lambda u: f'{u}[0]')
        reg('invrho', (), lambda u: f'1.0/{u}[0]')
        reg('E', (), lambda u: f'{u}[{nvars - 1}]')
        reg('v', ('invrho',),
            lambda u: [f'invrho*{u}[{i + 1}]' for i in range(ndims)])
        reg('p', ('invrho', 'E'),
            lambda u: f'{gamma - 1}*(E - 0.5*invrho*('
            + ' + '.join(f'({u}[{i + 1}])*({u}[{i + 1}])'
                         for i in range(ndims))
            + '))')
        reg('a', ('rho', 'p'), lambda u: f'sqrt({gamma}*p/rho)')
        reg('cpT', ('invrho', 'E', 'v'),
            lambda u: f'{gamma}*(invrho*E - 0.5*('
            + ' + '.join(f'(v[{i}])*(v[{i}])' for i in range(ndims))
            + '))')

        # Physical entropy (clamped for non-physical states)
        fpmax = self.fpdtype_max
        reg('s', ('rho', 'invrho', 'p'),
            device=lambda u, sfx: (
                f'fpdtype_t s{sfx} = (rho{sfx} > 0 && p{sfx} > 0)'
                f' ? p{sfx}*pow(invrho{sfx}, {gamma}) : {fpmax};'
            ),
            host=lambda u, ns: np.where(
                (ns['rho'] > 0) & (ns['p'] > 0),
                ns['p']*np.power(ns['invrho'], gamma), fpmax
            ))

    @property
    def privars(self):
        if self.ndims == 2:
            return ['rho', 'u', 'v', 'p']
        else:
            return ['rho', 'u', 'v', 'w', 'p']

    @property
    def convars(self):
        if self.ndims == 2:
            return ['rho', 'rhou', 'rhov', 'E']
        else:
            return ['rho', 'rhou', 'rhov', 'rhow', 'E']

    @property
    def dualcoeffs(self):
        return self.convars

    @property
    def visvars(self):
        if self.ndims == 2:
            return {
                'density': ['rho'],
                'velocity': ['u', 'v'],
                'pressure': ['p']
            }
        else:
            return {
                'density': ['rho'],
                'velocity': ['u', 'v', 'w'],
                'pressure': ['p']
            }

    def pri_seed(self, pris):
        return {'rho': pris[0], 'v': list(pris[1:-1]), 'p': pris[-1]}

    def con_to_pri(self, cons):
        q = self.eval('v, p', cons)

        return [cons[0], *q['v'], q['p']]

    def pri_to_con(self, pris):
        rho, p = pris[0], pris[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in pris[1:-1]]

        # Compute the energy
        E = p/(self.c['gamma'] - 1) + 0.5*rho*sum(c*c for c in pris[1:-1])

        return [rho, *rhovs, E]

    def grad_con_to_pri(self, cons, grad_cons):
        rho, *rhouvw = cons[:-1]
        grad_rho, *grad_rhouvw, grad_E = grad_cons

        # Divide momentum components by ρ
        uvw = [rhov / rho for rhov in rhouvw]

        # Velocity gradients: ∇u⃗ = 1/ρ·[∇(ρu⃗) - u⃗ ⊗ ∇ρ]
        grad_uvw = [(grad_rhov - v*grad_rho) / rho
                    for grad_rhov, v in zip(grad_rhouvw, uvw)]

        # Pressure gradient: ∇p = (γ - 1)·[∇E - 1/2*(u⃗·∇(ρu⃗) - ρu⃗·∇u⃗)]
        grad_p = grad_E - 0.5*(np.einsum('ijk,iljk->ljk', uvw, grad_rhouvw) +
                               np.einsum('ijk,iljk->ljk', rhouvw, grad_uvw))
        grad_p *= (self.c['gamma'] - 1)

        return [grad_rho, *grad_uvw, grad_p]

    def diff_con_to_pri(self, cons, diff_cons):
        rho, *rhouvw = cons[:-1]
        diff_rho, *diff_rhouvw, diff_E = diff_cons

        # Divide momentum components by ρ
        uvw = [rhov / rho for rhov in rhouvw]

        # Velocity gradients: ∂u⃗ = 1/ρ·[∂(ρu⃗) - u⃗·∂ρ]
        diff_uvw = [(diff_rhov - v*diff_rho) / rho
                    for diff_rhov, v in zip(diff_rhouvw, uvw)]

        # Pressure gradient: ∂p = (γ - 1)·[∂E - 1/2*(u⃗·∂(ρu⃗) + ρu⃗·∂u⃗)]
        diff_p = diff_E - 0.5*(sum(u*dru for u, dru in zip(uvw, diff_rhouvw)) +
                               sum(ru*du for ru, du in zip(rhouvw, diff_uvw)))
        diff_p *= self.c['gamma'] - 1

        return [diff_rho, *diff_uvw, diff_p]
