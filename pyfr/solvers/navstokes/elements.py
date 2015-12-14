# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base.kernels import ComputeMetaKernel
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements
from pyfr.util import ndrange


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

            backend.pointwise.register('pyfr.solvers.navstokes.kernels.entropy')
            backend.pointwise.register('pyfr.solvers.navstokes.kernels.avis')

            def artf_vis():
                # Compute entropy and save to avis_upts
                ent = backend.kernel(
                    'entropy', tplargs=tplargs, dims=[nupts, neles],
                    u=self.scal_upts_inb, s=avis_upts
                )

                # Compute modal coefficients of entropy
                rcpvdm = np.linalg.inv(self.basis.ubasis.vdm.T)
                rcpvdm = self._be.const_matrix(rcpvdm, tags={'align'})
                mul = backend.kernel(
                    'mul', rcpvdm, avis_upts, out=avis_upts_temp
                )

                # Additional constants for element-wise artificial viscosity
                tplargs['c'].update(
                    self.cfg.items_as('solver-artificial-viscosity', float)
                )
                tplargs.update(dict(
                    nupts=nupts, nfpts=nfpts, order=self.basis.order,
                    ubdegs=self.basis.ubasis.degrees,
                ))

                # Column view for avis_upts/fpts matrices
                col_view = lambda mat: backend.view(
                    matmap=np.array([mat.mid]*mat.ncol),
                    rcmap=np.array(list(ndrange(1, mat.ncol))),
                    stridemap=np.array([[mat.leaddim]]*mat.ncol),
                    vshape=(mat.nrow,)
                )

                avis_fpts_cv = col_view(self._avis_fpts)
                avis_upts_temp_cv = col_view(avis_upts_temp)

                if 'flux' in self.antialias:
                    ame_e = col_view(avis_qpts)
                    tplargs['nrow_amu'] = self.nqpts
                else:
                    ame_e = col_view(avis_upts)
                    tplargs['nrow_amu'] = nupts

                # Element-wise artificial viscosity kernel
                avis = backend.kernel(
                    'avis', tplargs, dims=[neles], s=avis_upts_temp_cv,
                    amu_e=ame_e, amu_f=avis_fpts_cv
                )

                return ComputeMetaKernel([ent, mul, avis])

            self.kernels['avis'] = artf_vis
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
