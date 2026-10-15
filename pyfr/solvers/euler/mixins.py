from collections import namedtuple

from pyfr.readers.native import Connectivity

import numpy as np


_NSCBCFace = namedtuple('_NSCBCFace', ('lhs', 'tplargs', 'extern_args',
                                       'extern_vals', 'kdata'))


class NSCBCMixin:
    flip_norm = False
    _nscbc_kern = 'pyfr.solvers.euler.kernels.bccflux_nscbc'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # NSCBC replaces comm_flux with its own corrected flux, scheduled
        # after the other interface fluxes
        self.kernels.pop('comm_flux', None)
        self._be.pointwise.register(self._nscbc_kern)
        self.kernels['nscbc_flux'] = lambda: self.gen_nscbc_kerns()

        # One kernel is emitted per element-face pair on the boundary
        self.nscbc_faces = []
        for etype, fidx, eidxs, idx in lhs.foreach():
            lhs_efp = Connectivity(lhs.cidxs[idx], lhs.eidxs[idx], lhs.cidxmap)
            self.nscbc_faces.append(self._build_face(etype, fidx, lhs_efp))

    def _build_face(self, etype, fidx, lhs):
        ele = self.elemap[etype]
        basis = ele.basis
        ndims = self.ndims
        nupts, nfpts = basis.nupts, basis.nfpts
        nfacefpts = basis.nfacefpts[fidx]
        facefpts = basis.facefpts[fidx]
        allfpts = [i for ff in basis.facefpts for i in ff]
        intfpts = [i for i in allfpts if i not in facefpts]

        # The transformed flux picks up a viscous part for Navier-Stokes
        visc = hasattr(ele, '_grad_upts')

        # Unit reference normals at the face flux points
        magn = np.linalg.norm(basis.norm_fpts, axis=-1)
        norm_ref = basis.norm_fpts[facefpts] / magn[facefpts, None]
        if self.flip_norm:
            norm_ref = -norm_ref

        # Correction-function divergences used to invert the FR update
        gbasis_fpts = basis.gbasis_at(basis.fpts)
        GB = gbasis_fpts[np.ix_(facefpts, facefpts)]
        GI = gbasis_fpts[np.ix_(facefpts, intfpts)]
        GB_inv = np.linalg.inv(GB)

        # Element geometry restricted to this pair's elements
        eidxs = lhs.eidxs
        smats = ele.smat_at_np('upts')[..., eidxs]
        smats = smats.transpose(1, 0, 2, 3).reshape(nupts, ndims*ndims, -1)
        jacs = 1.0 / ele.rcpdjac_at_np('fpts')[facefpts][:, eidxs]

        # The transformed-flux evaluation reads the copied solution at upts
        ele._soln_copy_required = True

        jac_fpts = np.rollaxis(basis.ubasis.jac_nodal_basis_at(basis.fpts), 2)
        m0 = basis.m0[facefpts]
        m2 = basis.m2.reshape(nfpts, ndims*nupts)
        m_div = jac_fpts[facefpts].reshape(nfacefpts, ndims*nupts)

        kdata = dict(
            u_upts=self._ewise_view(etype, eidxs, '_scal_upts_cpy',
                                    (nupts, self.nvars)),
            u_fpts=self._ewise_view(etype, eidxs, '_scal_fpts',
                                    (nfpts, self.nvars)),
            smats_upts=self._be.const_matrix(smats),
            jacs_ffpts=self._be.const_matrix(jacs),
            m0=self._be.const_matrix(m0),
            m2=self._be.const_matrix(m2),
            m_div=self._be.const_matrix(m_div)
        )
        if visc:
            kdata['gradu_upts'] = self._ewise_view(etype, eidxs, '_grad_upts',
                                                   (ndims*nupts, self.nvars))

        tplargs = self._tplargs | dict(
            nupts=nupts, nfpts=nfpts, nfacefpts=nfacefpts,
            facefpts=facefpts, intfpts=intfpts, norm_ref=norm_ref,
            GB_inv=GB_inv, GB_inv_GI=GB_inv @ GI, waves=self.waves
        )

        return _NSCBCFace(
            lhs=lhs, tplargs=tplargs,
            extern_args=self._external_args.copy(),
            extern_vals=self._external_vals.copy(),
            kdata=kdata
        )

    def _ewise_view(self, etype, eidxs, buf, vshape):
        mat = getattr(self.elemap[etype], buf)
        n = len(eidxs)
        return self._be.view(np.full(n, mat.mid), np.zeros(n, dtype=int),
                             eidxs, vshape=vshape)

    def gen_nscbc_kerns(self):
        kerns = [
            self._be.kernel('bccflux_nscbc', tplargs=f.tplargs,
                            dims=[len(f.lhs)], extrns=f.extern_args,
                            **f.kdata, **f.extern_vals)
            for f in self.nscbc_faces
        ]

        return self._be.unordered_meta_kernel(kerns)
