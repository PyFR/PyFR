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
        tdisf = []
        c, l = 'curved', 'linear'
        r, s = self._mesh_regions, self._slice_mat

        # Gradient + flux kernel fusion
        if self.grad_fusion:
            if c in r:
                tdisf.append(lambda uin: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'curved-fused'},
                    dims=[self.nupts, r[c]],  u=s(self.scal_upts[uin], c),
                    f=s(self._vect_upts, c), gradu=s(self._grad_upts, c),
                    rcpdjac=self.rcpdjac_at('upts', 'curved'),
                    smats=self.curved_smat_at('upts')
                ))
            if l in r:
                tdisf.append(lambda uin: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'linear-fused'},
                    dims=[self.nupts, r[l]], u=s(self.scal_upts[uin], l),
                    f=s(self._vect_upts, l), gradu=s(self._grad_upts, l),
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
                    f=s(self._vect_qpts, c), smats=self.curved_smat_at('qpts')
                ))
            if l in r:
                tdisf.append(lambda: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'linear'},
                    dims=[self.nqpts, r[l]], u=s(self._scal_qpts, l),
                    f=s(self._vect_qpts, l), verts=self.ploc_at('linspts', l),
                    upts=self.qpts
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
                    f=s(self._vect_upts, c), smats=self.curved_smat_at('upts')
                ))
            if l in r:
                tdisf.append(lambda uin: self._be.kernel(
                    'tflux', tplargs=tplargs | {'ktype': 'linear'},
                    dims=[self.nupts, r[l]], u=s(self.scal_upts[uin], l),
                    f=s(self._vect_upts, l), verts=self.ploc_at('linspts', l),
                    upts=self.upts
                ))

            def tdisf_k(uin):
                return self._make_sliced_kernel(k(uin) for k in tdisf)

            self.kernels['tdisf'] = tdisf_k
