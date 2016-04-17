# -*- coding: utf-8 -*-

import numpy as np

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    def set_backend(self, backend, nscalupts):
        super().set_backend(backend, nscalupts)
        backend.pointwise.register('pyfr.solvers.navstokes.kernels.tflux')

        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        if visc_corr not in {'sutherland', 'none'}:
            raise ValueError('Invalid viscosity-correction option')

        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                       visc_corr=visc_corr,
                       c=self.cfg.items_as('constants', float))

        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'artificial-viscosity':
            # Allocate required scratch space for artificial viscosity
            nupts, nfpts, neles = self.nupts, self.nfpts, self.neles
            tags = {'align'}

            if 'flux' in self.antialias:
                avis_qpts = backend.matrix((self.nqpts, 1, neles),
                                           extent='avis_qpts', tags=tags)
                avis_upts = backend.matrix((nupts, 1, neles),
                                           aliases=avis_qpts, tags=tags)
            else:
                avis_upts = backend.matrix((nupts, 1, neles),
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

            # Entropy kernel
            backend.pointwise.register(
                'pyfr.solvers.navstokes.kernels.entropy'
            )

            self.kernels['entropy'] = lambda: backend.kernel(
                'entropy', tplargs=tplargs, dims=[nupts, neles],
                u=self.scal_upts_inb, s=avis_upts
            )

            # Modal entropy coefficient kernel
            rcpvdm = np.linalg.inv(self.basis.ubasis.vdm.T)
            rcpvdm = self._be.const_matrix(rcpvdm, tags={'align'})
            self.kernels['modal_entropy'] = lambda: backend.kernel(
                'mul', rcpvdm, avis_upts, out=avis_upts_temp
            )

            # Artificial viscosity kernel
            backend.pointwise.register('pyfr.solvers.navstokes.kernels.avis')
            art_visc_tplargs = dict(
                tplargs, nupts=nupts, nfpts=nfpts, order=self.basis.order,
                ubdegs=self.basis.ubasis.degrees
            )
            art_visc_tplargs['c'].update(
                self.cfg.items_as('solver-artificial-viscosity', float)
            )

            if 'flux' in self.antialias:
                ame_e = avis_qpts
                art_visc_tplargs['nrow_amu'] = self.nqpts
            else:
                ame_e = avis_upts
                art_visc_tplargs['nrow_amu'] = nupts

            # Element-wise artificial viscosity kernel
            self.kernels['art_visc'] = lambda: backend.kernel(
                'avis', tplargs=art_visc_tplargs, dims=[neles],
                s=avis_upts_temp, amu_e=ame_e, amu_f=self._avis_fpts
            )

            tplargs['art_vis'] = 'mu'
        elif shock_capturing == 'none':
            avis_upts = avis_qpts = None
            tplargs['art_vis'] = 'none'
        else:
            raise ValueError('Invalid shock-capturing option')

        if 'flux' in self.antialias:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self.smat_at('qpts'),
                f=self._vect_qpts, amu=avis_qpts
            )
        else:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self.smat_at('upts'),
                f=self._vect_upts, amu=avis_upts
            )
