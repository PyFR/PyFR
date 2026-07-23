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

        reg('rho', (), lambda u: f'{u}[0]',
            diff=lambda u, du: f'{du}[0]')
        reg('invrho', (), lambda u: f'1.0/{u}[0]')
        reg('E', (), lambda u: f'{u}[{nvars - 1}]')
        reg('v', ('invrho',),
            lambda u: [f'invrho*{u}[{i + 1}]' for i in range(ndims)],
            diff=lambda u, du: [f'({du}[{i + 1}] - v[{i}]*d_rho)/rho'
                                for i in range(ndims)])
        reg('p', ('invrho', 'E'),
            lambda u: f'{gamma - 1}*(E - 0.5*invrho*('
            + ' + '.join(f'({u}[{i + 1}])*({u}[{i + 1}])'
                         for i in range(ndims))
            + '))',
            diff=lambda u, du: f'{gamma - 1}*({du}[{nvars - 1}] - 0.5*(('
            + ' + '.join(f'v[{i}]*{du}[{i + 1}]' for i in range(ndims))
            + ') + ('
            + ' + '.join(f'{u}[{i + 1}]*d_v[{i}]' for i in range(ndims))
            + ')))')
        reg('a', ('rho', 'p'), lambda u: f'sqrt({gamma}*p/rho)',
            diff=lambda u, du: '0.5*a*(d_p/p - d_rho/rho)')
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

    @property
    def pri_map(self):
        return [('rho', None),
                *[('v', i) for i in range(self.ndims)],
                ('p', None)]

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

