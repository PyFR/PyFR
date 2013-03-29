# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import sympy as sy

from pyfr.nputil import npeval
from pyfr.util import ndrange, lazyprop


class BaseAdvectionElements(object):
    __metaclass__ = ABCMeta

    # Map from dimension number to list of dynamical variables
    _dynvarmap = None

    _nscal_fpts = 1
    _nvect_upts = 1
    _nvect_fpts = 0

    def __init__(self, basiscls, eles, cfg):
        self._be = None
        self._cfg = cfg

        self.nspts = nspts = eles.shape[0]
        self.neles = neles = eles.shape[1]
        self.ndims = ndims = eles.shape[2]

        # Subclass checks
        assert self._dynvarmap
        assert self._nscal_fpts >= 1
        assert self._nvect_upts >= 1
        assert self._nvect_fpts >= 0

        # Check the dimensionality of the problem
        if ndims != basiscls.ndims or ndims not in self._dynvarmap:
            raise ValueError('Invalid element matrix dimensions')

        # Determine the number of dynamical variables
        self.nvars = len(self._dynvarmap[ndims])

        # Generate a symbol for each dimension (p,q or p,q,r)
        dims = sy.symbols('p q r')[:ndims]

        # Instantiate the basis class
        self._basis = basis = basiscls(dims, nspts, cfg)

        # Sizes
        self.nupts = nupts = basis.nupts
        self.nfpts = nfpts = sum(basis.nfpts)

        # Transform matrices at the soln points
        self._gen_rcpdjac_smat_upts(eles)

        # Physical normals at the flux points
        self._gen_pnorm_fpts(eles)

        # Physical locations of the solution points
        self._gen_ploc_upts(eles)

    def _process_ics(self, ics):
        return ics

    def set_ics_from_expr(self):
        nupts, neles, ndims = self._ploc_upts.shape

        # Bring simulation constants into scope
        vars = self._cfg.items_as('constants', float)

        if any(d in vars for d in 'xyz'):
            raise ValueError('Invalid constants (x, y, or z) in config file')

        # Extract the components of the mesh coordinates
        coords = np.rollaxis(self._ploc_upts, 2)
        vars.update(dict(zip('xyz', coords)))

        # Evaluate the ICs from the config file
        ics = [npeval(self._cfg.get('mesh-ics', dv), vars)
               for dv in self._dynvarmap[ndims]]

        # Allow subclasses to process these ICs
        ics = np.dstack(self._process_ics(ics))

        # Handle the case of uniform (scalar) ICs
        if ics.shape[:2] == (1, 1):
            ics = ics*np.ones((nupts, neles, 1))

        self._scal_upts = ics

    def set_ics_from_soln(self, solnmat, solncfg):
        # Recreate the existing solution basis
        currb = self._basis
        solnb = currb.__class__(currb._dims, None, solncfg)

        # Form the interpolation operator
        interp = solnb.ubasis_at(currb.upts)
        interp = np.asanyarray(interp, dtype=np.float)

        # Sizes
        nupts, neles, nvars = self.nupts, self.neles, self.nvars

        # Apply and reshape
        self._scal_upts = np.dot(interp, solnmat.reshape(solnb.nupts, -1))
        self._scal_upts = self._scal_upts.reshape(nupts, neles, nvars)

    def set_backend(self, be, nscal_upts):
        # Ensure a backend has not already been set
        assert self._be is None
        self._be = be

        # Allocate the constant operator matrices
        self._m0b = be.auto_matrix(self.m0, tags={'M0'})
        self._m3b = be.auto_matrix(self.m3, tags={'M3'})
        self._m132b = be.auto_matrix(self.m132, tags={'M132'})

        # Allocate soln point transformation matrices
        self._rcpdjac_upts = be.const_matrix(self._rcpdjac_upts)
        self._smat_upts = be.const_matrix(self._smat_upts)

        # Sizes
        nupts, nfpts = self.nupts, self.nfpts
        nvars, ndims = self.nvars, self.ndims
        neles = self.neles

        # Tags for flux matrices
        tags = {'align'}

        # Allocate general storage required for flux computation
        self._scal_upts = [be.matrix((nupts, neles, nvars), self._scal_upts,
                                     tags=tags)
                           for i in xrange(nscal_upts)]
        self._vect_upts = [be.matrix((nupts, ndims, neles, nvars), tags=tags)
                           for i in xrange(self._nvect_upts)]
        self._scal_fpts = [be.matrix((nfpts, neles, nvars), tags=tags)
                           for i in xrange(self._nscal_fpts)]
        self._vect_fpts = [be.matrix((nfpts, ndims, neles, nvars), tags=tags)
                           for i in xrange(self._nvect_fpts)]

        # Bank the scalar soln points (as required by the RK schemes)
        self.scal_upts_inb = be.matrix_bank(self._scal_upts)
        self.scal_upts_outb = be.matrix_bank(self._scal_upts)

        # Pre-compute some of the matrices required for constructing views
        # onto scal_fpts and pnorm_fpts
        self._gen_inter_view_mats(be, neles, nvars, ndims)

    def _gen_inter_view_mats(self, be, neles, nvars, ndims):
        # Get the number of flux points for each face of the element
        self._nfacefpts = nfacefpts = self._basis.nfpts

        # Get the relevant strides required for view construction
        self._scal_fpts_strides = be.from_aos_stride_to_native(neles, nvars)

        # Pre-compute for the max flux point count on a given face
        nmaxfpts = max(nfacefpts)

        # View stride info (common to all scal_fpts mats)
        self._scal_fpts_vstri = np.empty((1, nmaxfpts), dtype=np.int32)
        self._scal_fpts_vstri[:] = self._scal_fpts_strides[1]

        # View matrix info
        self._scal_fpts_vmats = [np.tile(m, (1, nmaxfpts))
                                 for m in self._scal_fpts]

    def get_scal_upts_mat(self, idx):
        return self._scal_upts[idx].get()

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
        return self._be.kernel('negdivconf', self.nvars,
                               self.scal_upts_outb, self._rcpdjac_upts)

    @lazyprop
    def m0(self):
        """Discontinuous soln at upts to discontinuous soln at fpts"""
        return self._basis.ubasis_at(self._basis.fpts)

    @lazyprop
    def m1(self):
        """Trans discontinuous flux at upts to trans divergence of
        trans discontinuous flux at upts
        """
        return self._basis.jac_ubasis_at(self._basis.upts)

    @lazyprop
    def m2(self):
        """Trans discontinuous flux at upts to trans normal
        discontinuous flux at fpts
        """
        return self._basis.norm_fpts[:,None,:]*self.m0[...,None]

    @lazyprop
    def m3(self):
        """Trans normal correction flux at upts to trans divergence of
        trans correction flux at upts
        """
        return self._basis.fbasis_at(self._basis.upts)

    @property
    def m132(self):
        m1, m2, m3 = self.m1, self.m2, self.m3
        return m1 - np.dot(m3, m2.reshape(self.nfpts, -1)).reshape(m1.shape)

    def _gen_rcpdjac_smat_upts(self, eles):
        jacs = self._get_jac_eles_at(eles, self._basis.upts)
        smats, djacs = self._get_smats(jacs, retdets=True)

        neles, ndims = eles.shape[1:]

        self._rcpdjac_upts = 1.0 / djacs.reshape(-1, neles)
        self._smat_upts = smats.reshape(-1, neles, ndims**2)

    def _gen_ploc_upts(self, eles):
        nspts, neles, ndims = eles.shape
        nupts = self.nupts

        # Construct the interpolation matrix
        op = np.asanyarray(self._basis.sbasis_at(self._basis.upts),
                           dtype=np.float)

        # Apply the operator and reshape
        self._ploc_upts = np.dot(op, eles.reshape(nspts, -1))\
                            .reshape(nupts, neles, ndims)

    def _gen_pnorm_fpts(self, eles):
        jac = self._get_jac_eles_at(eles, self._basis.fpts)
        smats = self._get_smats(jac)

        normfpts = np.asanyarray(self._basis.norm_fpts, dtype=np.float)

        # Reshape (nfpts*neles, ndims, dims) => (nfpts, neles, ndims, ndims)
        smats = smats.reshape(self.nfpts, -1, self.ndims, self.ndims)

        # We need to compute |J|*[(J^{-1})^{T}.N] where J is the
        # Jacobian and N is the normal for each fpt.  Using
        # J^{-1} = S/|J| where S are the smats, we have S^{T}.N.
        pnorm_fpts = np.einsum('ijlk,il->ijk', smats, normfpts)

        # Compute the magnitudes of these flux point normals
        mag_pnorm_fpts = np.einsum('...i,...i', pnorm_fpts, pnorm_fpts)
        mag_pnorm_fpts = np.sqrt(mag_pnorm_fpts)

        # Normalize the physical normals at the flux points
        self._norm_pnorm_fpts = pnorm_fpts / mag_pnorm_fpts[...,None]
        self._mag_pnorm_fpts = mag_pnorm_fpts

    def _get_jac_eles_at(self, eles, pts):
        nspts, neles, ndims = eles.shape
        npts = len(pts)

        # Form the Jacobian operator (going from AoS to SoA)
        jacop = self._basis.jac_sbasis_at(pts).swapaxes(1, 2)

        # Convert to double precision
        jacop = np.asanyarray(jacop, dtype=np.float)

        # Cast as a matrix multiply and apply to eles
        jac = np.dot(jacop.reshape(-1, nspts), eles.reshape(nspts, -1))

        # Reshape (npts*ndims, neles*ndims) => (npts, ndims, neles, ndims)
        jac = jac.reshape(npts, ndims, neles, ndims)

        # Transpose to get (npts, neles, ndims, ndims) ≅ (npts, neles, J)
        jac = jac.transpose(0, 2, 3, 1)

        return jac.reshape(-1, ndims, ndims)

    def _get_smats(self, jac, retdets=False):
        if self.ndims == 2:
            return self._get_smats2d(jac, retdets)
        elif self.ndims == 3:
            return self._get_smats3d(jac, retdets)
        else:
            raise ValueError('Invalid basis dimension')

    def _get_smats2d(self, jac, retdets):
        a, b, c, d = [jac[:,i,j] for i, j in ndrange(2,2)]

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

    def get_mag_pnorms_for_inter(self, eidx, fidx, rtag):
        fpts_idx = self._basis.fpts_idx_for_face(fidx, rtag)
        return self._mag_pnorm_fpts[fpts_idx, eidx]

    def get_norm_pnorms_for_inter(self, eidx, fidx, rtag):
        fpts_idx = self._basis.fpts_idx_for_face(fidx, rtag)
        return self._norm_pnorm_fpts[fpts_idx, eidx]

    def _get_scal_fptsn_for_inter(self, n, eidx, fidx, rtag):
        nfp = self._nfacefpts[fidx]

        vrcidx = np.empty((1, nfp, 2), dtype=np.int32)
        vrcidx[...,0] = self._basis.fpts_idx_for_face(fidx, rtag)
        vrcidx[...,1] = eidx*self._scal_fpts_strides[0]

        return (self._scal_fpts_vmats[n][:nfp], vrcidx,
                self._scal_fpts_vstri[:nfp])

    def _get_vect_fptsn_for_inter(self, n, eidx, fidx, rtag):
        nfp = self._nfacefpts[fidx]

        vrcidx = np.empty((self.ndims, nfp, 2), dtype=np.int32)
        vrcidx[...,0] = self._basis.fpts_idx_for_face(fidx, rtag)
        vrcidx[...,1] = eidx*self._scal_fpts_strides[0]

        # Correct the row indicies
        for i in range(self.ndims):
            vrcidx[i,:,0] += i*self.nfpts

        return (self._vect_fpts_vmats[n][:nfp], vrcidx,
                self._vect_fpts_vstri[:nfp])

    def get_scal_fpts0_for_inter(self, eidx, fidx, rtag):
        return self._get_scal_fptsn_for_inter(0, eidx, fidx, rtag)


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
        self._m5b = be.auto_matrix(self.m5, tags={'M5'})
        self._m6b = be.auto_matrix(self.m6, tags={'M6'})
        self._m460b = be.auto_matrix(self.m460, tags={'M460'})

        # Flux point transformation matrices
        self._jmat_fpts = be.const_matrix(self._jmat_fpts)

    def _gen_inter_view_mats(self, be, neles, nvars, ndims):
        base = super(BaseAdvectionDiffusionElements, self)._gen_inter_view_mats
        base(be, neles, nvars, ndims)

        # Vector-view stride info
        self._vect_fpts_vstri = np.tile(self._scal_fpts_vstri, (self.ndims, 1))

        # Vector view matrix info
        self._vect_fpts_vmats = [np.tile(m, self._vect_fpts_vstri.shape)
                                 for m in self._vect_fpts]

    @lazyprop
    def m4(self):
        """Discontinuous soln at upts to trans gradient of discontinuous
        solution at upts
        """
        return self.m1.swapaxes(2, 1)[...,None]

    @lazyprop
    def m5(self):
        """Trans grad discontinuous soln at upts to trans gradient of
        discontinuous solution at fpts
        """
        nfpts, ndims, nupts = self.nfpts, self.ndims, self.nupts
        m = np.zeros((nfpts, ndims, nupts, ndims), dtype=self.m0.dtype)

        for i in xrange(ndims):
            m[:,i,:,i] = self.m0

        return m

    @lazyprop
    def m6(self):
        """Correction soln at fpts to trans gradient of correction
        solution at upts
        """
        m = self._basis.norm_fpts.T[:,None,:]*self.m3
        return m.swapaxes(0, 1)[...,None]

    @property
    def m460(self):
        m4, m6, m0 = self.m4, self.m6, self.m0
        return m4 - np.dot(m6.reshape(-1, self.nfpts), m0).reshape(m4.shape)

    def _gen_jmats_fpts(self, eles):
        jac = self._get_jac_eles_at(eles, self._basis.fpts)
        smats, djacs = self._get_smats(jac, retdets=True)

        # Use J^-1 = S/|J| hence J^-T = S^T/|J|
        jmat_fpts = smats.swapaxes(1, 2) / djacs[...,None,None]

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
        return self._be.kernel('gradcoru', self.ndims, self.nvars,
                               self._jmat_fpts, self._vect_fpts[0])

    def get_scal_fpts1_for_inter(self, eidx, fidx, rtag):
        return self._get_scal_fptsn_for_inter(1, eidx, fidx, rtag)

    def get_vect_fpts0_for_inter(self, eidx, fidx, rtag):
        return self._get_vect_fptsn_for_inter(0, eidx, fidx, rtag)


class BaseFluidElements(object):
    _dynvarmap = {2: ['rho', 'u', 'v', 'p'],
                  3: ['rho', 'u', 'v', 'w', 'p']}

    def _process_ics(self, ics):
        rho, p = ics[0], ics[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in ics[1:-1]]

        # Compute the energy
        gamma = self._cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in ics[1:-1])

        return [rho] + rhovs + [E]


class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def get_tdisf_upts_kern(self):
        kc = self._cfg.items_as('constants', float)

        return self._be.kernel('tdisf_inv', self.ndims, self.nvars,
                               self.scal_upts_inb, self._smat_upts,
                               self._vect_upts[0], kc)


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    def get_tdisf_upts_kern(self):
        kc = self._cfg.items_as('constants', float)

        return self._be.kernel('tdisf_vis', self.ndims, self.nvars,
                               self.scal_upts_inb, self._smat_upts,
                               self._rcpdjac_upts, self._vect_upts[0], kc)
