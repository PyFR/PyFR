# -*- coding: utf-8 -*-

import numpy as np

from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    _nscal_fpts = 2
    _nvect_upts = 1
    _nvect_fpts = 1

    def __init__(self, basiscls, eles, cfg):
        super(BaseAdvectionDiffusionElements, self).__init__(basiscls, eles,
                                                             cfg)

        self._gen_jmats_fpts(eles)

    def set_backend(self, be, nscal_upts):
        super(BaseAdvectionDiffusionElements, self).set_backend(be, nscal_upts)

        # Allocate the additional operator matrices
        self._m5b = be.auto_matrix(self._basis.m5, tags={'M5'})
        self._m6b = be.auto_matrix(self._basis.m6, tags={'M6'})
        self._m460b = be.auto_matrix(self._basis.m460, tags={'M460'})

        # Flux point transformation matrices
        self._jmat_fpts = be.const_matrix(self._jmat_fpts, tags={'align'})

        # Vector-view stride info
        self._vect_fpts_vstri = np.tile(self._scal_fpts_vstri, (self.ndims, 1))

        # Vector view matrix info
        self._vect_fpts_vmats = [np.tile(m, self._vect_fpts_vstri.shape)
                                 for m in self._vect_fpts]

        # Register pointwise kernels
        be.pointwise.register('pyfr.solvers.baseadvecdiff.kernels.gradcoru')

    def _gen_jmats_fpts(self, eles):
        jac = self._get_jac_eles_at(eles, self._basis.fpts)
        smats, djacs = self._get_smats(jac, retdets=True)

        # Use J^-1 = S/|J| hence J^-T = S^T/|J|
        jmat_fpts = smats.swapaxes(2, 3) / djacs[...,None,None]

        self._jmat_fpts = jmat_fpts.reshape(self.nfpts, -1, self.ndims**2)

    def get_tgradpcoru_upts_kern(self):
        return self._be.kernel('mul', self._m460b, self.scal_upts_inb,
                               out=self._vect_upts[0])

    def get_tgradcoru_upts_kern(self):
        return self._be.kernel('mul', self._m6b, self._scal_fpts[1],
                               out=self._vect_upts[0], beta=1.0)

    def get_tgradcoru_fpts_kern(self):
        return self._be.kernel('mul', self._m5b, self._vect_upts[0],
                               out=self._vect_fpts[0])

    def get_gradcoru_fpts_kern(self):
        tplargs = dict(ndims=self.ndims, nvars=self.nvars)

        return self._be.kernel('gradcoru', tplargs,
                               dims=[self.nfpts, self.neles],
                               jmats=self._jmat_fpts, gradu=self._vect_fpts[0])

    def get_scal_fpts1_for_inter(self, eidx, fidx, rtag):
        return self._get_scal_fptsn_for_inter(1, eidx, fidx, rtag)

    def get_vect_fpts0_for_inter(self, eidx, fidx, rtag):
        return self._get_vect_fptsn_for_inter(0, eidx, fidx, rtag)
