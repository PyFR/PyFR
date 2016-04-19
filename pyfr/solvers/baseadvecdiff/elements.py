# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base.kernels import ComputeMetaKernel
from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    @property
    def _scratch_bufs(self):
        bufs = {'scal_fpts', 'vect_fpts', 'vect_upts'}

        if 'flux' in self.antialias:
            bufs |= {'scal_qpts', 'vect_qpts'}
        elif 'div-flux' in self.antialias:
            bufs |= {'scal_qpts'}

        if self._soln_in_src_exprs:
            if 'div-flux' in self.antialias:
                bufs |= {'scal_qpts_cpy'}
            else:
                bufs |= {'scal_upts_cpy'}

        return bufs

    def _set_backend_art_visc(self, backend):
        nvars, neles = self.nvars, self.neles
        nupts, nfpts = self.nupts, self.nfpts
        tags = {'align'}

        # Register pointwise kernels
        backend.pointwise.register(
            'pyfr.solvers.baseadvecdiff.kernels.shockvar'
        )
        backend.pointwise.register(
            'pyfr.solvers.baseadvecdiff.kernels.shocksensor'
        )

        # Obtain the scalar variable to be used for shock sensing
        shockvar = self.convarmap[self.ndims].index(self.shockvar)

        # Common template arguments
        tplargs = dict(
            nvars=nvars, nupts=nupts, nfpts=nfpts,
            c=self.cfg.items_as('solver-artificial-viscosity', float),
            order=self.basis.order, ubdegs=self.basis.ubasis.degrees,
            shockvar=shockvar
        )

        # Allocate space for the artificial viscosity
        self.avis = backend.matrix((1, neles), extent='avis', tags=tags)

        # Scratch space
        tmp_upts = backend.matrix((2*nupts, 1, neles),
                                  aliases=self._vect_upts, tags=tags)
        svar_upts = tmp_upts.rslice(0, nupts)
        modal_svar_upts = tmp_upts.rslice(nupts, 2*nupts)


        # Extract the scalar variable to be used for shock sensing
        self.kernels['shockvar'] = lambda: backend.kernel(
            'shockvar', tplargs=tplargs, dims=[nupts, neles],
            u=self.scal_upts_inb, s=svar_upts
        )

        # Obtain the modal coefficients
        rcpvdm = np.linalg.inv(self.basis.ubasis.vdm.T)
        rcpvdm = backend.const_matrix(rcpvdm, tags={'align'})
        self.kernels['shockvar_modal'] = lambda: backend.kernel(
            'mul', rcpvdm, svar_upts, out=modal_svar_upts
        )

        # Apply the sensor to estimate the required artificial viscosity
        self.kernels['shocksensor'] = lambda: backend.kernel(
            'shocksensor', tplargs=tplargs, dims=[neles],
            s=modal_svar_upts, amu=self.avis
        )

    def set_backend(self, backend, nscal_upts):
        super().set_backend(backend, nscal_upts)

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
                    for i in range(self.ndims)]

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
                        for i in range(self.ndims)]

                return ComputeMetaKernel(muls)

            self.kernels['gradcoru_qpts'] = gradcoru_qpts

        # Shock capturing
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'artificial-viscosity':
            self._set_backend_art_visc(backend)
        elif shock_capturing == 'none':
            self.avis = None
        else:
            raise ValueError('Invalid shock capturing scheme')

    def get_avis_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]
        return (self.avis.mid,)*nfp, ((0, eidx),)*nfp
