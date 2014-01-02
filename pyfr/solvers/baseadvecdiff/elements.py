# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements
from pyfr.backends.base.kernels import ComputeMetaKernel


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    _nscal_fpts = 1
    _nvect_upts = 1
    _nvect_fpts = 1

    def set_backend(self, be, nscal_upts):
        super(BaseAdvectionDiffusionElements, self).set_backend(be, nscal_upts)

        # Allocate the additional operator matrices
        self._m6b = be.const_matrix(self._basis.m6, tags={'M6'})
        self._m460b = be.const_matrix(self._basis.m460, tags={'M460'})

        # Register pointwise kernels
        be.pointwise.register('pyfr.solvers.baseadvecdiff.kernels.gradcoru')

    def get_tgradpcoru_upts_kern(self):
        return self._be.kernel('mul', self._m460b, self.scal_upts_inb,
                               out=self._vect_upts[0])

    def get_tgradcoru_upts_kern(self):
        scal_fpts1 = self._vect_fpts[0].rslice(0, self.nfpts)
        return self._be.kernel('mul', self._m6b, scal_fpts1,
                               out=self._vect_upts[0], beta=1.0)

    def get_gradcoru_upts_kern(self):
        tplargs = dict(ndims=self.ndims, nvars=self.nvars)

        return self._be.kernel('gradcoru', tplargs,
                               dims=[self.nupts, self.neles],
                               smats=self._smat_upts,
                               rcpdjac=self._rcpdjac_upts,
                               gradu=self._vect_upts[0])

    def get_gradcoru_fpts_kern(self):
        nupts, nfpts = self.nupts, self.nfpts
        vect_upts, vect_fpts = self._vect_upts[0], self._vect_fpts[0]

        # Here we exploit the fact that the operator to move gradients
        # at the solution points to the flux points is diag([m0]*ndims)
        # and so can cast it as ndim applications of m0
        muls = [self._be.kernel('mul', self._m0b,
                                vect_upts.rslice(i*nupts, (i + 1)*nupts),
                                vect_fpts.rslice(i*nfpts, (i + 1)*nfpts))
                for i in xrange(self.ndims)]

        return ComputeMetaKernel(muls)

    def get_vect_fpts0_for_inter(self, eidx, fidx):
        return self._get_vect_fptsn_for_inter(self._vect_fpts[0], eidx, fidx)
