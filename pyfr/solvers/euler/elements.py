# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements
import numpy as np

class BaseFluidElements:
    formulations = ['std', 'dual']

    privarmap = {2: ['rho', 'u', 'v', 'p'],
                 3: ['rho', 'u', 'v', 'w', 'p']}

    convarmap = {2: ['rho', 'rhou', 'rhov', 'E'],
                 3: ['rho', 'rhou', 'rhov', 'rhow', 'E']}

    dualcoeffs = convarmap

    visvarmap = {
        2: [('density', ['rho']),
            ('velocity', ['u', 'v']),
            ('pressure', ['p'])],
        3: [('density', ['rho']),
            ('velocity', ['u', 'v', 'w']),
            ('pressure', ['p'])]
    }

    @staticmethod
    def pri_to_con(pris, cfg):
        rho, p = pris[0], pris[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in pris[1:-1]]

        # Compute the energy
        gamma = cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in pris[1:-1])

        return [rho] + rhovs + [E]

    @staticmethod
    def con_to_pri(cons, cfg):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]


class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Can elide interior flux calculations at p = 0
        if self.basis.order == 0:
            return

        # Register our flux kernels
        self._be.pointwise.register('pyfr.solvers.euler.kernels.tflux')
        self._be.pointwise.register('pyfr.solvers.euler.kernels.tfluxlin')
        self._be.pointwise.register('pyfr.solvers.euler.kernels.entropylocal')
        self._be.pointwise.register('pyfr.solvers.euler.kernels.entropyfilter')

        # Template parameters for the flux kernels
        tplargs = {
            'ndims': self.ndims,
            'nupts': self.nupts,
            'nfpts': self.nfpts,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs
        }

        # Helpers
        c, l = 'curved', 'linear'
        r, s = self._mesh_regions, self._slice_mat

        if c in r and 'flux' not in self.antialias:
            self.kernels['tdisf_curved'] = lambda uin: self._be.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, r[c]],
                u=s(self.scal_upts[uin], c), f=s(self._vect_upts, c),
                smats=self.curved_smat_at('upts')
            )
        elif c in r:
            self.kernels['tdisf_curved'] = lambda: self._be.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, r[c]],
                u=s(self._scal_qpts, c), f=s(self._vect_qpts, c),
                smats=self.curved_smat_at('qpts')
            )

        if l in r and 'flux' not in self.antialias:
            self.kernels['tdisf_linear'] = lambda uin: self._be.kernel(
                'tfluxlin', tplargs=tplargs, dims=[self.nupts, r[l]],
                u=s(self.scal_upts[uin], l), f=s(self._vect_upts, l),
                verts=self.ploc_at('linspts', l), upts=self.upts
            )
        elif l in r:
            self.kernels['tdisf_linear'] = lambda: self._be.kernel(
                'tfluxlin', tplargs=tplargs, dims=[self.nqpts, r[l]],
                u=s(self._scal_qpts, l), f=s(self._vect_qpts, l),
                verts=self.ploc_at('linspts', l), upts=self.qpts
            )

        if self.cfg.get('solver', 'shock-capturing') == 'entropy-filter':
            # Minimum density/pressure constraints
            d_min = float(self.cfg.get('solver-entropy-filter', 'd_min', 1e-6))
            p_min = float(self.cfg.get('solver-entropy-filter', 'p_min', 1e-6))
            # Absolute entropy tolerance
            e_atol = float(self.cfg.get('solver-entropy-filter', 'e_atol', 1e-6))
            # Relative entropy tolerance (with respect to maximum variation in entropy within element)
            e_rtol = float(self.cfg.get('solver-entropy-filter', 'e_rtol', 1e-3))
            # Number of iterations to compute filter strength
            niters = int(self.cfg.get('solver-entropy-filter', 'niters', 10))
            # Maximum filter strength
            precision = self.cfg.get('backend', 'precision')
            zeta_max = -np.log(1e-7) if precision == 'single' else -np.log(1e-16)

            # See if applying constraints to fpts/qpts
            con_fpts = self.cfg.getbool('solver-entropy-filter', 'constrain-fpts', False)
            con_qpts = self.cfg.getbool('solver-entropy-filter', 'constrain-qpts', False)
            nqpts = self.nqpts if self.nqpts else 1

            # Precompute basis orders for filter
            ubdegs2 = [max(dd)**2 for dd in self.basis.ubasis.degrees]

            eftplargs = {
                'ndims': self.ndims, 'nupts': self.nupts, 'nfpts': self.nfpts,
                'nqpts': nqpts, 'nvars': self.nvars,
                'c': self.cfg.items_as('constants', float),
                'd_min': d_min, 'p_min': p_min, 'e_atol': e_atol,
                'e_rtol': e_rtol, 'niters': niters, 'zeta_max': zeta_max,
                'con_fpts': con_fpts, 'con_qpts': con_qpts, 'ubdegs2': ubdegs2,
            }

            # Compute local entropy bounds
            self.kernels['local_entropy'] = lambda uin: self._be.kernel(
                'entropylocal', tplargs=eftplargs, dims=[self.neles],
                u=self.scal_upts[uin], entmin=self.entmin, 
                entmin_int=self.entmin_int
            )

            # Apply entropy filter
            self.kernels['filter_solution'] = lambda uin: self._be.kernel(
                'entropyfilter', tplargs=eftplargs, dims=[self.neles],
                u=self.scal_upts[uin], entmin=self.entmin,
                vdm=self.vdm, invvdm=self.invvdm,
                intfpts=self.intfpts, intqpts=self.intqpts
            )
