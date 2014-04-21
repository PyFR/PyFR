# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements
from pyfr.backends.base.kernels import ComputeMetaKernel


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    _need_vect_fpts = True

    def set_backend(self, backend, nscal_upts):
        super(BaseAdvectionDiffusionElements, self).set_backend(backend,
                                                                nscal_upts)

        # Register pointwise kernels
        backend.pointwise.register(
            'pyfr.solvers.baseadvecdiff.kernels.gradcoru'
        )

        self._m460b = backend.const_matrix(self._basis.m460, tags={'M460'})
        self.kernels['tgradpcoru_upts'] = lambda: backend.kernel(
            'mul', self._m460b, self.scal_upts_inb, out=self._vect_upts
        )

        self._m6b = backend.const_matrix(self._basis.m6, tags={'M6'})
        self.kernels['tgradcoru_upts'] = lambda: backend.kernel(
            'mul', self._m6b, self._vect_fpts.rslice(0, self.nfpts),
             out=self._vect_upts, beta=1.0
        )

        self.kernels['gradcoru_upts'] = lambda: backend.kernel(
            'gradcoru', tplargs=dict(ndims=self.ndims, nvars=self.nvars),
             dims=[self.nupts, self.neles], smats=self._smat_upts,
             rcpdjac=self._rcpdjac_upts, gradu=self._vect_upts
        )

        def gradcoru_fpts():
            nupts, nfpts = self.nupts, self.nfpts
            vect_upts, vect_fpts = self._vect_upts, self._vect_fpts

            # Exploit the block-diagonal form of the operator
            muls = [backend.kernel('mul', self._m0b,
                                   vect_upts.rslice(i*nupts, (i + 1)*nupts),
                                   vect_fpts.rslice(i*nfpts, (i + 1)*nfpts))
                    for i in xrange(self.ndims)]

            return ComputeMetaKernel(muls)

        self.kernels['gradcoru_fpts'] = gradcoru_fpts
