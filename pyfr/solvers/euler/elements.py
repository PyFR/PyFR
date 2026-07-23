import numpy as np

from pyfr.fluids import get_fluid
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
            'jac_exprs': self.basis.jac_exprs,
            'fluid': get_fluid(self.cfg, self.ndims)
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
        return get_fluid(cfg, ndims).privars

    @staticmethod
    def convars(ndims, cfg):
        return get_fluid(cfg, ndims).convars

    dualcoeffs = convars

    @staticmethod
    def visvars(ndims, cfg):
        return get_fluid(cfg, ndims).visvars

    @staticmethod
    def pri_to_con(pris, cfg):
        return get_fluid(cfg, len(pris) - 2).pri_to_con(pris)

    @staticmethod
    def con_to_pri(cons, cfg):
        return get_fluid(cfg, len(cons) - 2).con_to_pri(cons)


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
            'jac_exprs': self.basis.jac_exprs,
            'fluid': get_fluid(self.cfg, self.ndims)
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
