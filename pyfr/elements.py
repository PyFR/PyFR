# -*- coding: utf-8 -*-

import numpy as np
import sympy as sy

from sympy.utilities.lambdify import lambdify

from pyfr.util import ndrange, npeval

class Elements(object):
    def __init__(self, basiscls, eles, cfg):
        self._be = None
        self._cfg = cfg

        self.nspts = nspts = eles.shape[0]
        self.neles = neles = eles.shape[1]
        self.ndims = ndims = eles.shape[2]
        self.nvars = nvars = ndims + 2

        # Checks
        if ndims != basiscls.ndims:
            raise ValueError('Invalid element matrix dimensions')

        # Generate a symbol for each dimension (p,q or p,q,r)
        dims = sy.symbols('p q r')[:ndims]

        # Instantiate the basis class
        self._basis = basis = basiscls(dims, cfg)

        # Get the locations of the soln, flux and shape points
        upts, ubasis = basis.get_upts()
        fpts, fbasis = basis.get_fpts()
        spts, sbasis = basis.get_spts(nspts)

        # Get the normals to the flux points
        normfpts = basis.get_norm_fpts()

        # Sizes
        self.nupts = nupts = len(upts)
        self.nfpts = nfpts = len(fpts)

        # Generate the constant operator matrices
        self._gen_m0(dims, ubasis, fpts)
        self._gen_m1(dims, ubasis, upts)
        self._gen_m2(self._m0, normfpts)
        self._gen_m3(dims, fbasis, upts)

        # Transform matrices at the soln points
        self._gen_rcpdjac_smat_upts(dims, eles, sbasis, upts)

        # Physical normals at the flux points
        self._gen_pnorm_fpts(dims, eles, sbasis, fpts, normfpts)

        # Physical locations of the solution points
        self._gen_ploc_upts(dims, eles, sbasis, upts)

    def set_ics(self):
        nupts, neles, ndims = self._ploc_upts.shape

        # Extract the individual coordinates
        coords = dict(zip(['x', 'y', 'z'], self._ploc_upts.transpose(2, 0, 1)))

        # Determine the dynamical variables in the simulation
        if self.nvars == 5:
            dvars = ('rho', 'rhou', 'rhov', 'rhow', 'E')
        else:
            dvars = ('rho', 'rhou', 'rhov', 'E')

        self._scal_upts = ics_upts = np.empty((nupts, neles, len(dvars)))
        for i,v in enumerate(dvars):
            ics_upts[...,i] = npeval(self._cfg.get('mesh-ics', v), coords)

    def set_backend(self, be, nsubanks):
        # Ensure a backend has not already been set
        assert self._be is None
        self._be = be

        # Pack higher dimensional matrices down into two dimensions
        self._m1 = be.from_aos_to_native(self._m1)
        self._m2 = be.from_aos_to_native(self._m2)
        self._smat_upts = be.from_aos_to_native(self._smat_upts)
        self._pnorm_fpts = be.from_aos_to_native(self._pnorm_fpts)
        self._scal_upts = be.from_aos_to_native(self._scal_upts)

        # Allocate the constant operator matrices
        self._m0 = be.auto_const_sparse_matrix(self._m0, tags={'M0'})
        self._m1 = be.auto_const_sparse_matrix(self._m1, tags={'M1'})
        self._m2 = be.auto_const_sparse_matrix(self._m2, tags={'M2'})
        self._m3 = be.auto_const_sparse_matrix(self._m3, tags={'M3'})

        # Allocate soln point transformation matrices
        self._rcpdjac_upts = be.const_matrix(self._rcpdjac_upts)
        self._smat_upts = be.const_matrix(self._smat_upts)

        # Allocate the physical normals at the flux points
        self._pnorm_fpts = be.const_matrix(self._pnorm_fpts)

        # Sizes
        nupts, nfpts = self.nupts, self.nfpts
        nvars, ndims = self.nvars, self.ndims
        neles = self.neles

        # Allocate general storage required for flux computation
        self._scal_upts = [be.matrix(nupts, nvars*neles, self._scal_upts)
                           for i in xrange(nsubanks)]
        self._vect_upts = be.matrix(nupts*ndims, nvars*neles)
        self._scal_fpts = be.matrix(nfpts, nvars*neles)

        # Bank the scalar soln points (as required by the RK schemes)
        self.scal_upts_inb = be.matrix_bank(self._scal_upts)
        self.scal_upts_outb = be.matrix_bank(self._scal_upts)

        # TODO: Contract m2 m3

        # Pre-compute some of the matrices required for constructing views
        # onto scal_fpts and pnorm_fpts
        self._gen_inter_view_mats(be, neles, nvars, ndims)

    def _gen_inter_view_mats(self, be, neles, nvars, ndims):
        # Get the number of flux points for each face of the element
        self._nfacefpts = nfacefpts = self._basis.get_nfpts()

        # Get the relevant strides required for view construction
        self._scal_fpts_strides = be.from_aos_stride_to_native(neles, nvars)
        self._pnorm_fpts_strides = be.from_aos_stride_to_native(neles, ndims)

        # Pre-compute for the max flux point count on a given face
        nmaxfpts = max(nfacefpts)
        self._scal_fpts_vmats = np.empty(nmaxfpts, dtype=np.object)
        self._scal_fpts_vstri = np.empty(nmaxfpts, dtype=np.int32)
        self._scal_fpts_vmats[:] = self._scal_fpts
        self._scal_fpts_vstri[:] = self._scal_fpts_strides[1]

        self._pnorm_fpts_vmats = np.empty(nmaxfpts, dtype=np.object)
        self._pnorm_fpts_vstri = np.empty(nmaxfpts, dtype=np.int32)
        self._pnorm_fpts_vmats[:] = self._pnorm_fpts
        self._pnorm_fpts_vstri[:] = self._pnorm_fpts_strides[1]

    def get_disu_fpts_kern(self):
        disu_upts, disu_fpts = self.scal_upts_inb, self._scal_fpts
        return self._be.kernel('mul', self._m0, disu_upts, disu_fpts)

    def get_tdisf_upts_kern(self):
        # User-defined constant
        gamma = self._cfg.getfloat('constants', 'gamma')

        # Element specific constant data
        smats_upts = self._smat_upts

        # Input solutions and output fluxes
        disu_upts, tdisf_upts = self.scal_upts_inb, self._vect_upts
        smats = self._smat_upts

        return self._be.kernel('tdisf_inv', self.ndims, self.nvars,
                               disu_upts, smats, tdisf_upts, gamma)

    def get_divtdisf_upts_kern(self):
        tdisf_upts, divtconf_upts = self._vect_upts, self.scal_upts_outb
        return self._be.kernel('mul', self._m1, tdisf_upts, divtconf_upts)

    def get_nrmtdisf_fpts_kern(self):
        tdisf_upts, normtcorf_fpts = self._vect_upts, self._scal_fpts
        return self._be.kernel('mul', self._m2, tdisf_upts, normtcorf_fpts,
                               beta=1.0, alpha=-1.0)

    def get_tdivtconf_upts_kern(self):
        normtcorf_fpts, tdivtconf_upts = self._scal_fpts, self.scal_upts_outb
        return self._be.kernel('mul', self._m3, normtcorf_fpts, tdivtconf_upts,
                               beta=1.0)

    def get_divconf_upts_kern(self):
        tdivtconf_upts, rcpdjac_upts = self.scal_upts_outb, self._rcpdjac_upts
        return self._be.kernel('divconf', self.ndims, self.nvars,
                               tdivtconf_upts, rcpdjac_upts)

    def _gen_m0(self, dims, ubasis, fpts):
        """Discontinuous soln at upts to discontinuous soln at fpts"""
        self._m0 = m = np.empty((len(fpts), len(ubasis)))

        ubasis_l = [lambdify(dims, u) for u in ubasis]
        for i,j in ndrange(*m.shape):
            m[i,j] = ubasis_l[j](*fpts[i])

    def _gen_m1(self, dims, ubasis, upts):
        """Trans discontinuous flux at upts to trans divergence of
        trans discontinuous flux at upts
        """
        self._m1 = m = np.empty((len(upts), len(upts), len(dims)))

        for j,k in ndrange(len(upts), len(dims)):
            p = lambdify(dims, ubasis[j].diff(dims[k]))
            for i in xrange(len(upts)):
                m[i,j,k] = p(*upts[i])

    def _gen_m2(self, m0, normfpts):
        """Trans discontinuous flux at upts to trans normal
        discontinuous flux at fpts
        """
        self._m2 = normfpts[:,None,:]*m0[...,None]

    def _gen_m3(self, dims, fbasis, upts):
        """Trans normal correction flux at upts to trans divergence of
        trans correction flux at upts
        """
        self._m3 = m = np.empty((len(upts), len(fbasis)))

        fbasis_l = [lambdify(dims, f) for f in fbasis]
        for i,j in ndrange(*m.shape):
            m[i,j] = fbasis_l[j](*upts[i])

    def _gen_rcpdjac_smat_upts(self, dims, eles, sbasis, upts):
        jacs = self._get_jac_eles(dims, eles, sbasis, upts)
        smats, djacs = self._get_smats(jacs, retdets=True)

        neles, ndims = eles.shape[1:]

        self._rcpdjac_upts = 1.0 / djacs.reshape(-1, neles)
        self._smat_upts = smats.reshape(-1, neles, ndims**2)

    def _gen_ploc_upts(self, dims, eles, sbasis, upts):
        nspts, neles, ndims = eles.shape
        nupts = len(upts)

        # Form the operator matrix
        op = np.empty((nupts, nspts))
        for j in xrange(nspts):
            sbasis_l = lambdify(dims, sbasis[j])
            for i in xrange(nupts):
                op[i,j] = sbasis_l(*upts[i])

        # Apply the operator and reshape
        self._ploc_upts = np.dot(op, eles.reshape(nspts, -1))\
                            .reshape(nupts, neles, ndims)

    def _gen_pnorm_fpts(self, dims, eles, sbasis, fpts, normfpts):
        jac = self._get_jac_eles(dims, eles, sbasis, fpts)
        smats = self._get_smats(jac)

        # Reshape (nfpts*neles, ndims, dims) => (nfpts, neles, ndims, ndims)
        smats = smats.reshape(len(fpts), -1, len(dims), len(dims))

        # We need to compute |J|*[(J^{-1})^{T}.N] where J is the
        # Jacobian and N is the normal for each fpt.  Using
        # J^{-1} = S/|J| where S are the smats, we have S^{T}.N.
        self._pnorm_fpts = np.einsum('ijlk,il->ijk', smats, normfpts)

    def _get_jac_eles(self, dims, eles, basis, pts):
        nspts, neles, ndims = eles.shape
        npts = len(pts)

        # Form the Jacobian operator (c.f, _gen_m2)
        jacop = np.empty((npts, ndims, nspts))
        for j,k in ndrange(ndims, nspts):
            dm = lambdify(dims, basis[k].diff(dims[j]))
            for i in xrange(npts):
                jacop[i,j,k] = dm(*pts[i])

        # Cast as a matrix multiply and apply to eles
        jac = np.dot(jacop.reshape(-1, nspts), eles.reshape(nspts, -1))

        # Reshape (npts*ndims, neles*ndims) => (npts, ndims, neles, ndims)
        jac = jac.reshape(npts, ndims, neles, ndims)

        # Transpose to get (npts, neles, ndims, ndims) ≅ (npts, neles, J)
        jac = jac.transpose(0, 2, 1, 3)

        return jac.reshape(-1, ndims, ndims)

    def _get_smats(self, jac, retdets=False):
        if self.ndims == 2:
            return self._get_smats2d(jac, retdets)
        elif self.ndims == 3:
            return self._get_smats3d(jac, retdets)
        else:
            raise ValueError('Invalid basis dimension')

    def _get_smats2d(self, jac, retdets):
        a, b, c, d = [jac[:,i,j] for i,j in ndrange(2,2)]

        smats = np.empty_like(jac)
        smats[:,0,0], smats[:,0,1] =  d, -b
        smats[:,1,0], smats[:,1,1] = -c,  a

        if retdets:
            return smats, a*c - b*d
        else:
            return smats

    def _get_smats3d(self, jac, retdets):
        smats = np.empty_like(jac)
        smats[:,0,:] = np.cross(jac[:,:,1], jac[:,:,2])
        smats[:,1,:] = np.cross(jac[:,:,2], jac[:,:,0])
        smats[:,2,:] = np.cross(jac[:,:,0], jac[:,:,1])

        if retdets:
            # Exploit the fact that det(J) = x0 . (x1 ^ x2); J = [x0, x1, x2]
            djacs = np.einsum('...i,...i', jac[:,:,0], smats[:,0,:])

            return smats, djacs
        else:
            return smats

    def get_pnorm_fpts_for_inter(self, eidx, fidx, rtag):
        n = self._nfacefpts[fidx]

        vrcidx = np.empty((n, 2), dtype=np.int32)
        vrcidx[:,0] = self._basis.get_fpts_for_face(fidx, rtag)
        vrcidx[:,1] = eidx*self._pnorm_fpts_strides[0]

        return self._pnorm_fpts_vmats[:n], vrcidx, self._pnorm_fpts_vstri[:n]

    def get_scal_fpts_for_inter(self, eidx, fidx, rtag):
        n = self._nfacefpts[fidx]

        vrcidx = np.empty((n, 2), dtype=np.int32)
        vrcidx[:,0] = self._basis.get_fpts_for_face(fidx, rtag)
        vrcidx[:,1] = eidx*self._scal_fpts_strides[0]

        return self._scal_fpts_vmats[:n], vrcidx, self._scal_fpts_vstri[:n]
