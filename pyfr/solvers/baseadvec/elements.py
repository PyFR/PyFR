# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseElements


class BaseAdvectionElements(BaseElements):
    def set_backend(self, be, nscal_upts):
        super(BaseAdvectionElements, self).set_backend(be, nscal_upts)

        # Get the number of flux points for each face of the element
        self.nfacefpts = nfacefpts = self._basis.nfacefpts

        # Register pointwise kernels with the backend
        be.pointwise.register('pyfr.solvers.baseadvec.kernels.negdivconf')

        m0b, m132b, m3b = self._m0b, self._m132b, self._m3b

        # Specify the kernels we provide
        self.kernels['disu_fpts'] = lambda: be.kernel(
            'mul', m0b, self.scal_upts_inb, out=self._scal_fpts[0]
        )
        self.kernels['tdivtpcorf_upts'] = lambda: be.kernel(
            'mul', m132b, self._vect_upts[0], out=self.scal_upts_outb
        )
        self.kernels['tdivtconf_upts'] = lambda: be.kernel(
            'mul', m3b, self._scal_fpts[0], out=self.scal_upts_outb, beta=1.0
        )
        self.kernels['negdivconf_upts'] = lambda: be.kernel(
            'negdivconf', tplargs=dict(nvars=self.nvars),
            dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
            rcpdjac=self._rcpdjac_upts
        )

    def get_mag_pnorms_for_inter(self, eidx, fidx):
        fpts_idx = self._srtd_face_fpts[fidx][eidx]
        return self._mag_pnorm_fpts[fpts_idx,eidx]

    def get_norm_pnorms_for_inter(self, eidx, fidx):
        fpts_idx = self._srtd_face_fpts[fidx][eidx]
        return self._norm_pnorm_fpts[fpts_idx,eidx]

    def _get_scal_fptsn_for_inter(self, mat, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[fidx][eidx]]
        cstri = [(mat.leadsubdim,)]*nfp

        return [mat]*nfp, rcmap, cstri

    def _get_vect_fptsn_for_inter(self, mat, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[fidx][eidx]]
        rcstri = [(self.nfpts, mat.leadsubdim)]*nfp

        return [mat]*nfp, rcmap, rcstri

    def get_scal_fpts0_for_inter(self, eidx, fidx):
        return self._get_scal_fptsn_for_inter(self._scal_fpts[0], eidx, fidx)
