from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseACFluidElements:
    @staticmethod
    def privars(ndims, cfg):
        return ['p', 'u', 'v'] if ndims == 2 else ['p', 'u', 'v', 'w']

    convars = privars

    @staticmethod
    def dualcoeffs(ndims, cfg):
        return ['u', 'v'] if ndims == 2 else ['u', 'v', 'w']

    @staticmethod
    def visvars(ndims, cfg):
        if ndims == 2:
            return {
                'velocity': ['u', 'v'],
                'pressure': ['p']
            }
        elif ndims == 3:
            return {
                'velocity': ['u', 'v', 'w'],
                'pressure': ['p']
            }

    @staticmethod
    def pri_to_con(pris, cfg):
        return list(pris)

    @staticmethod
    def con_to_pri(convs, cfg):
        return list(convs)

    @staticmethod
    def diff_con_to_pri(cons, diff_cons, cfg):
        return list(diff_cons)

    @staticmethod
    def validate_formulation(controller):
        if controller.formulation != 'dual':
            raise ValueError('System not compatible with time stepping '
                             'formulation.')


class ACEulerElements(BaseACFluidElements, BaseAdvectionElements):
    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Can elide interior flux calculations at p = 0
        if self.basis.order == 0:
            return

        # Register our flux kernels
        self._be.pointwise.register('pyfr.solvers.aceuler.kernels.tflux')

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
        r, s = self._mesh_regions, self._slice_mat
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
                dims=[self.nqpts, r[c]],
                u=s(self._scal_qpts, c), f=s(self._vect_qpts, c),
                smats=self.curved_smat_at('qpts')
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

        self.kernels['tdisf'] = lambda: slicedk(k() for k in tdisf)
