# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import sympy as sy

from pyfr.nputil import npeval
from pyfr.util import ndrange


class BaseElements(object):
    __metaclass__ = ABCMeta

    # Map from dimension number to list of dynamical variables
    _dynvarmap = None

    _nscal_fpts = 1
    _nvect_upts = 1
    _nvect_fpts = 0

    def __init__(self, basiscls, eles, cfg):
        self._be = None

        self._eles = eles
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
        self.nfpts = nfpts = basis.nfpts

        # Transform matrices at the soln points
        self._gen_rcpdjac_smat_upts(eles)

        # Physical normals at the flux points
        self._gen_pnorm_fpts(eles)

    @abstractmethod
    def _process_ics(self, ics):
        pass

    def set_ics_from_expr(self):
        # Bring simulation constants into scope
        vars = self._cfg.items_as('constants', float)

        if any(d in vars for d in 'xyz'):
            raise ValueError('Invalid constants (x, y, or z) in config file')

        # Construct the physical location operator matrix
        plocop = np.asanyarray(self._basis.sbasis_at(self._basis.upts),
                               dtype=np.float)

        # Apply the operator to the mesh elements and reshape
        plocupts = np.dot(plocop, self._eles.reshape(self.nspts, -1))
        plocupts = plocupts.reshape(self.nupts, self.neles, self.ndims)

        # Extract the components of the mesh coordinates
        coords = np.rollaxis(plocupts, 2)
        vars.update(dict(zip('xyz', coords)))

        # Evaluate the ICs from the config file
        ics = [npeval(self._cfg.get('soln-ics', dv), vars)
               for dv in self._dynvarmap[self.ndims]]

        # Allow subclasses to process these ICs
        ics = np.dstack(self._process_ics(ics))

        # Handle the case of uniform (scalar) ICs
        if ics.shape[:2] == (1, 1):
            ics = ics*np.ones((self.nupts, self.neles, 1))

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

    @abstractmethod
    def set_backend(self, be, nscal_upts):
        # Ensure a backend has not already been set
        assert self._be is None
        self._be = be

        # Allocate the constant operator matrices
        self._m0b = be.auto_matrix(self._basis.m0, tags={'M0'})
        self._m3b = be.auto_matrix(self._basis.m3, tags={'M3'})
        self._m132b = be.auto_matrix(self._basis.m132, tags={'M132'})

        # Tags to ensure alignment of multi-dimensional matrices
        tags = {'align'}

        # Allocate soln point transformation matrices
        self._rcpdjac_upts = be.const_matrix(self._rcpdjac_upts, tags=tags)
        self._smat_upts = be.const_matrix(self._smat_upts, tags=tags)

        # Sizes
        nupts, nfpts = self.nupts, self.nfpts
        nvars, ndims = self.nvars, self.ndims
        neles = self.neles

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

    def get_scal_upts_mat(self, idx):
        return self._scal_upts[idx].get()

    def _gen_rcpdjac_smat_upts(self, eles):
        jacs = self._get_jac_eles_at(eles, self._basis.upts)
        smats, djacs = self._get_smats(jacs, retdets=True)

        # Check for negative Jacobians
        if np.any(djacs < -1e-5):
            raise RuntimeError('Negative mesh Jacobians detected')

        neles, ndims = eles.shape[1:]

        self._rcpdjac_upts = 1.0 / djacs.reshape(-1, neles)
        self._smat_upts = smats.reshape(-1, neles, ndims**2)

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
        a, b, c, d = [jac[:,i,j] for i, j in ndrange(2, 2)]

        smats = np.empty_like(jac)
        smats[:,0,0], smats[:,0,1] =  d, -b
        smats[:,1,0], smats[:,1,1] = -c,  a

        if retdets:
            return smats, a*d - b*c
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
