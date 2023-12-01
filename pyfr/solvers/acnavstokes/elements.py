from pyfr.solvers.aceuler.elements import BaseACFluidElements
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements


class ACNavierStokesElements(BaseACFluidElements,
                             BaseAdvectionDiffusionElements):
    @staticmethod
    def grad_con_to_pri(cons, grad_cons, cfg):
        return grad_cons

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Can elide interior flux calculations at p = 0
        if self.basis.order == 0:
            return

        # Register our flux kernels
        kernel, kernels = self._be.kernel, self.kernels
        kprefix = 'pyfr.solvers.acnavstokes.kernels'
        self._be.pointwise.register(f'{kprefix}.tflux')

        # Template parameters for the flux kernels
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs
        }

        # Helpers
        c, l = 'curved', 'linear'
        r, s = self._mesh_regions, self._slice_mat

        if c in r and self.grad_fusion:
            kernels['tdisf_fused_curved'] = lambda uin: kernel(
                'tflux', tplargs=tplargs | {'ktype': 'curved-fused'},
                dims=[self.nupts, r[c]],  u=s(self.scal_upts[uin], c),
                f=s(self._vect_upts, c), gradu=s(self._grad_upts, c),
                rcpdjac=self.rcpdjac_at('upts', 'curved'),
                smats=self.curved_smat_at('upts')
            )
        elif c in r:
            kernels['tdisf_curved'] = lambda: kernel(
                'tflux', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nqpts, r[c]], u=s(self._scal_qpts, c),
                f=s(self._vect_qpts, c), smats=self.curved_smat_at('qpts')
            )

        if l in r and self.grad_fusion:
            kernels['tdisf_fused_linear'] = lambda uin: kernel(
                'tflux', tplargs=tplargs | {'ktype': 'linear-fused'},
                dims=[self.nupts, r[l]], u=s(self.scal_upts[uin], l),
                f=s(self._vect_upts, l), gradu=s(self._grad_upts, l),
                verts=self.ploc_at('linspts', l), upts=self.upts
            )
        elif l in r:
            kernels['tdisf_linear'] = lambda: kernel(
                'tflux', tplargs=tplargs | {'ktype': 'linear'},
                dims=[self.nqpts, r[l]], u=s(self._scal_qpts, l),
                f=s(self._vect_qpts, l), verts=self.ploc_at('linspts', l),
                upts=self.qpts
            )
