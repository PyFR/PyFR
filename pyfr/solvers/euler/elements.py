import numpy as np

from pyfr.solvers.base.elements import ExportableField
from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseFluidElements:
    eos_kernel_module = 'pyfr.solvers.euler.kernels.eos'

    @classmethod
    def eos_tplargs(cls, ndims, cfg):
        return {
            'ndims': ndims,
            'nvars': len(cls.convars(ndims, cfg)),
            'c': cfg.items_as('constants', float),
        }

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Register wavespeed kernel for CFL-based time stepping
        self._be.pointwise.register('pyfr.solvers.euler.kernels.wavespeed')

    def init_wavespeed(self):
        self._wspd = self._be.matrix((1, self.neles), tags={'align'})
        self.kernels['wavespeed'] = lambda uin: self._wavespeed_kernel(uin)

        def cfl_getter():
            with np.errstate(divide='ignore'):
                wspd = (2*self.basis.order + 1)*self._wspd.get()[0]
                return np.nan_to_num(1/wspd, posinf=0.0)

        self.export_fields.append(ExportableField(
            name='dt-cfl', shape=(), getter=cfl_getter
        ))
        return self._wspd

    def _wavespeed_kernel(self, uin):
        r, s = self.mesh_regions, self._slice_mat

        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs
        }

        wkerns = []
        for rgn in ('curved', 'linear'):
            if rgn not in r:
                continue

            if rgn == 'curved':
                kw = {'smats': self.curved_smat_at('upts'),
                      'rcpdjac': self.rcpdjac_at('upts', 'curved')}
            else:
                kw = {'verts': self.ploc_at('linspts', 'linear'),
                      'upts': self.upts}

            wkerns.append(self._be.kernel(
                'wavespeed', tplargs=tplargs | {'ktype': rgn},
                dims=[self.nupts, r[rgn]], u=s(self.scal_upts[uin], rgn),
                wspd=self._wspd, **kw
            ))

        if len(wkerns) > 1:
            return self._be.unordered_meta_kernel(wkerns)
        else:
            return wkerns[0]

    @staticmethod
    def privars(ndims, cfg):
        if ndims == 2:
            return ['rho', 'u', 'v', 'p']
        elif ndims == 3:
            return ['rho', 'u', 'v', 'w', 'p']

    @staticmethod
    def convars(ndims, cfg):
        if ndims == 2:
            return ['rho', 'rhou', 'rhov', 'E']
        elif ndims == 3:
            return ['rho', 'rhou', 'rhov', 'rhow', 'E']

    dualcoeffs = convars

    @staticmethod
    def visvars(ndims, cfg):
        if ndims == 2:
            return {
                'density': ['rho'],
                'velocity': ['u', 'v'],
                'pressure': ['p']
            }
        elif ndims == 3:
            return {
                'density': ['rho'],
                'velocity': ['u', 'v', 'w'],
                'pressure': ['p']
            }

    @staticmethod
    def pri_to_con(pris, cfg):
        rho, p = pris[0], pris[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in pris[1:-1]]

        # Compute the energy
        gamma = cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in pris[1:-1])

        return [rho, *rhovs, E]

    @staticmethod
    def con_to_pri(cons, cfg):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v**2 for v in vs))

        return [rho, *vs, p]

    @staticmethod
    def diff_con_to_pri(cons, diff_cons, cfg):
        rho, *rhouvw = cons[:-1]
        diff_rho, *diff_rhouvw, diff_E = diff_cons

        # Divide momentum components by ρ
        uvw = [rhov / rho for rhov in rhouvw]

        # Velocity gradients: ∂u⃗ = 1/ρ·[∂(ρu⃗) - u⃗·∂ρ]
        diff_uvw = [(diff_rhov - v*diff_rho) / rho
                    for diff_rhov, v in zip(diff_rhouvw, uvw)]

        # Pressure gradient: ∂p = (γ - 1)·[∂E - 1/2*(u⃗·∂(ρu⃗) + ρu⃗·∂u⃗)]
        gamma = cfg.getfloat('constants', 'gamma')
        diff_p = diff_E - 0.5*(sum(u*dru for u, dru in zip(uvw, diff_rhouvw)) +
                               sum(ru*du for ru, du in zip(rhouvw, diff_uvw)))
        diff_p *= gamma - 1

        return [diff_rho, *diff_uvw, diff_p]

class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Can elide interior flux calculations at p = 0
        if self.basis.order == 0:
            return

        # Register our flux kernels
        self._be.pointwise.register('pyfr.solvers.euler.kernels.tflux')

        # Template parameters for the flux kernels
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs
        }

        # Helpers
        tdisf = []
        c, l = 'curved', 'linear'
        r, s = self.mesh_regions, self._slice_mat
        slicedk = self._make_sliced_kernel

        if c in r and 'flux' not in self.antialias:
            tdisf.append(lambda uin: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nupts, r[c]], u=s(self.scal_upts[uin], c),
                f=s(self._vect_upts, c), smats=self.curved_smat_at('upts')
            ))
        elif c in r:
            tdisf.append(lambda: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nqpts, r[c]], u=s(self._scal_qpts, c),
                f=s(self._vect_qpts, c), smats=self.curved_smat_at('qpts')
            ))

        if l in r and 'flux' not in self.antialias:
            tdisf.append(lambda uin: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'linear'},
                dims=[self.nupts, r[l]], u=s(self.scal_upts[uin], l),
                f=s(self._vect_upts, l), verts=self.ploc_at('linspts', l),
                upts=self.upts
            ))
        elif l in r:
            tdisf.append(lambda: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'linear'},
                dims=[self.nqpts, r[l]], u=s(self._scal_qpts, l),
                f=s(self._vect_qpts, l), verts=self.ploc_at('linspts', l),
                upts=self.qpts
            ))

        if 'flux' not in self.antialias:
            self.kernels['tdisf'] = lambda uin: slicedk(k(uin) for k in tdisf)
        else:
            self.kernels['tdisf'] = lambda: slicedk(k() for k in tdisf)
