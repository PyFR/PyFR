# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements
from pyfr.backends.base.kernels import ComputeMetaKernel


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    @property
    def _scratch_bufs(self):
        comm = {'scal_fpts', 'vect_fpts', 'vect_upts'}
        if 'flux' in self.antialias:
            return comm | {'scal_qpts', 'vect_qpts'}
        elif 'div-flux' in self.antialias:
            return comm | {'scal_qpts'}
        else:
            return comm

    def set_backend(self, backend, nscal_upts):
        super(BaseAdvectionDiffusionElements, self).set_backend(backend,
                                                                nscal_upts)

        # Register pointwise kernels
        backend.pointwise.register(
            'pyfr.solvers.baseadvecdiff.kernels.gradcoru'
        )

        self.kernels['tgradpcoru_upts'] = lambda: backend.kernel(
            'mul', self.opmat('M4 - M6*M0'), self.scal_upts_inb,
            out=self._vect_upts
        )
        self.kernels['tgradcoru_upts'] = lambda: backend.kernel(
            'mul', self.opmat('M6'), self._vect_fpts.rslice(0, self.nfpts),
             out=self._vect_upts, beta=1.0
        )
        self.kernels['gradcoru_upts'] = lambda: backend.kernel(
            'gradcoru', tplargs=dict(ndims=self.ndims, nvars=self.nvars),
             dims=[self.nupts, self.neles], smats=self.smat_at('upts'),
             rcpdjac=self.rcpdjac_at('upts'), gradu=self._vect_upts
        )

        def gradcoru_fpts():
            nupts, nfpts = self.nupts, self.nfpts
            vupts, vfpts = self._vect_upts, self._vect_fpts

            # Exploit the block-diagonal form of the operator
            muls = [backend.kernel('mul', self.opmat('M0'),
                                   vupts.rslice(i*nupts, (i + 1)*nupts),
                                   vfpts.rslice(i*nfpts, (i + 1)*nfpts))
                    for i in xrange(self.ndims)]

            return ComputeMetaKernel(muls)

        self.kernels['gradcoru_fpts'] = gradcoru_fpts

        if 'flux' in self.antialias:
            def gradcoru_qpts():
                nupts, nqpts = self.nupts, self.nqpts
                vupts, vqpts = self._vect_upts, self._vect_qpts

                # Exploit the block-diagonal form of the operator
                muls = [backend.kernel('mul', self.opmat('M7'),
                                       vupts.rslice(i*nupts, (i + 1)*nupts),
                                       vqpts.rslice(i*nqpts, (i + 1)*nqpts))
                        for i in xrange(self.ndims)]

                return ComputeMetaKernel(muls)

            self.kernels['gradcoru_qpts'] = gradcoru_qpts
