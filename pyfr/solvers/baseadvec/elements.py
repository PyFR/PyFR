# -*- coding: utf-8 -*-

from abc import abstractmethod

import numpy as np

from pyfr.solvers.base import BaseElements


class BaseAdvectionElements(BaseElements):
    def set_backend(self, be, nscal_upts):
        super(BaseAdvectionElements, self).set_backend(be, nscal_upts)

        # Get the number of flux points for each face of the element
        self.nfacefpts = nfacefpts = self._basis.nfacefpts

        # Pre-compute for the max flux point count on a given face
        nmaxfpts = max(nfacefpts)

        # View and vector-view stride info
        self._scal_fpts_vstri = np.empty((1, nmaxfpts), dtype=np.int32)
        self._scal_fpts_vstri[:] = self._scal_fpts[0].leadsubdim
        self._vect_fpts_vstri = np.tile(self._scal_fpts_vstri, (self.ndims, 1))

        # Register pointwise kernels
        be.pointwise.register('pyfr.solvers.baseadvec.kernels.negdivconf')

    @abstractmethod
    def get_tdisf_upts_kern(self):
        pass

    def get_disu_fpts_kern(self):
        return self._be.kernel('mul', self._m0b, self.scal_upts_inb,
                               out=self._scal_fpts[0])

    def get_tdivtpcorf_upts_kern(self):
        return self._be.kernel('mul', self._m132b, self._vect_upts[0],
                               out=self.scal_upts_outb)

    def get_tdivtconf_upts_kern(self):
        return self._be.kernel('mul', self._m3b, self._scal_fpts[0],
                               out=self.scal_upts_outb, beta=1.0)

    def get_negdivconf_upts_kern(self):
        return self._be.kernel('negdivconf', tplargs=dict(nvars=self.nvars),
                               dims=[self.nupts, self.neles],
                               tdivtconf=self.scal_upts_outb,
                               rcpdjac=self._rcpdjac_upts)

    def get_mag_pnorms_for_inter(self, eidx, fidx):
        fpts_idx = self._srtd_face_fpts[eidx,fidx]
        return self._mag_pnorm_fpts[fpts_idx,eidx]

    def get_norm_pnorms_for_inter(self, eidx, fidx):
        fpts_idx = self._srtd_face_fpts[eidx,fidx]
        return self._norm_pnorm_fpts[fpts_idx,eidx]

    def _get_scal_fptsn_for_inter(self, mat, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[eidx,fidx]]
        cstri = [(mat.leadsubdim,)]*nfp

        return [mat]*nfp, rcmap, cstri

    def _get_vect_fptsn_for_inter(self, mat, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[eidx,fidx]]
        rcstri = [(self.nfpts, mat.leadsubdim)]*nfp

        return [mat]*nfp, rcmap, rcstri

    def get_scal_fpts0_for_inter(self, eidx, fidx):
        return self._get_scal_fptsn_for_inter(self._scal_fpts[0], eidx, fidx)
