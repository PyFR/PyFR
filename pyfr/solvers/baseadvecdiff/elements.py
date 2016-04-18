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

        # Allocate required scratch space for artificial viscosity
        if 'flux' in self.antialias:
            self.avis_qpts = backend.matrix((self.nqpts, 1, neles),
                                            extent='avis_qpts', tags=tags)
            self.avis_upts = backend.matrix((nupts, 1, neles),
                                            aliases=self.avis_qpts, tags=tags)
        else:
            self.avis_upts = backend.matrix((nupts, 1, neles),
                                             extent='avis_upts', tags=tags)

        if nfpts >= nupts:
            self._avis_fpts = backend.matrix((nfpts, 1, neles),
                                             extent='avis_fpts', tags=tags)
            avis_upts_temp = backend.matrix(
                (nupts, 1, neles), aliases=self._avis_fpts, tags=tags
            )
        else:
            avis_upts_temp = backend.matrix((nupts, 1, neles),
                                            extent='avis_fpts', tags=tags)
            self._avis_fpts = backend.matrix(
                (nfpts, 1, neles), aliases=avis_upts_temp, tags=tags
            )

        # Extract the scalar variable to be used for shock sensing
        self.kernels['shockvar'] = lambda: backend.kernel(
            'shockvar', tplargs=tplargs, dims=[nupts, neles],
            u=self.scal_upts_inb, s=self.avis_upts
        )

        # Obtain the modal coefficients
        rcpvdm = np.linalg.inv(self.basis.ubasis.vdm.T)
        rcpvdm = backend.const_matrix(rcpvdm, tags={'align'})
        self.kernels['shockvar_modal'] = lambda: backend.kernel(
            'mul', rcpvdm, self.avis_upts, out=avis_upts_temp
        )

        if 'flux' in self.antialias:
            ame_e = self.avis_qpts
            tplargs['nrow_amu'] = self.nqpts
        else:
            ame_e = self.avis_upts
            tplargs['nrow_amu'] = nupts

        # Apply the sensor to estimate the required artificial viscosity
        self.kernels['shocksensor'] = lambda: backend.kernel(
            'shocksensor', tplargs=tplargs, dims=[neles],
            s=avis_upts_temp, amu_e=ame_e, amu_f=self._avis_fpts
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
            self.avis_qpts = self.avis_upts = None
        else:
            raise ValueError('Invalid shock capturing scheme')

    def get_avis_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[fidx][eidx]]
        return (self._avis_fpts.mid,)*nfp, rcmap