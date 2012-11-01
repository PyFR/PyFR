# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

from pyfr.util import ndrange
from sympy.utilities.lambdify import lambdify

import numpy as np
import sympy as sy

class ElementsBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, be, eles, dims, nsubanks, cfg):
        self._be = be
        self._cfg = cfg
        self._order = order = int(cfg.get('scheme', 'order'))

        # Interrogate eles to get our dimensions
        nspts, neles = eles.shape[:2]
        ndims, nvars = len(dims), len(dims) + 2

        # Checks
        if eles.shape[2] != ndims:
            raise ValueError('Invalid element matrix dimensions')

        # Get the locations of the soln, flux and shape points
        upts, ubasis = self._gen_upts_basis(dims, order)
        fpts, fbasis = self._gen_fpts_basis(dims, order)
        spts, sbasis = self._gen_spts_basis(dims, nspts)

        # Get the normals to the flux points
        normfpts = self._gen_norm_fpts(dims, order)

        # Sizes
        nupts, nfpts = len(upts), len(fpts)

        # Generate the constant operator matrices
        m0 = self._gen_m0(dims, upts, ubasis, fpts)
        m1 = self._gen_m1(dims, upts, ubasis)
        m2 = self._gen_m2(dims, ubasis, fpts, normfpts)
        m3 = self._gen_m3(dims, upts, fbasis)

        djacupts, smatupts = self._gen_djac_smat_upts(dims, eles, sbasis, upts)
        transfpts = self._gen_trans_fpts(dims, eles, sbasis, fpts, normfpts)

        # Allocate these constant matrices on the backend
        self._m0 = be.auto_const_sparse_matrix(m0, tags={'M0'})
        self._m1 = be.auto_const_sparse_matrix(m1, tags={'M1'})
        self._m2 = be.auto_const_sparse_matrix(m2, tags={'M2'})
        self._m3 = be.auto_const_sparse_matrix(m3, tags={'M3'})

        self._djac_upts = be.const_matrix(djacupts)
        self._smat_upts = be.const_matrix(smatupts)
        self._trans_fpts = be.const_matrix(transfpts)

        # Allocate general storage required for flux computation
        self._scal_upts = [be.matrix(nupts, nvars*neles)
                           for i in xrange(nsubanks)]
        self._vect_upts = [be.matrix(nupts*ndims, nvars*neles)]
        self._scal_fpts = [be.matrix(nfpts, nvars*neles)]

        # Bank the scalar soln points (as required by the RK schemes)
        self._scal_upts_inb = be.matrix_bank(self._scal_upts)
        self._scal_upts_outb = be.matrix_bank(self._scal_upts)


    def gen_disu_fpts_kern(self):
        disu_upts, disu_fpts = self._scal_upts_inb, self._scal_fpts[0]
        return self._be.kernel('mul', self._m0, disu_upts, disu_fpts)

    def gen_tdisf_upts_kern(self):
        gamma = float(self._cfg.get('constants', 'gamma'))
        disu_upts, tdisf_upts = self._scal_upts_inb, self._vect_upts[0]
        return self._be.kernel('tdisf_upts', disu_upts, gamma, tdisf_upts)

    def gen_divtdisf_upts_kern(self):
        tdisf_upts, divtconf_upts = self._vect_upts[0], self._scal_upts_outb
        return self._be.kernel('mul', self._m1, tdisf_upts, divtconf_upts)

    def gen_nrmtdisf_fpts_kern(self):
        tdisf_upts, normtcorf_fpts = self._vect_upts[0], self._scal_fpts[0]
        return self._be.kernel('mul', self._m2, tdisf_upts, normtcorf_fpts)

    def gen_divtconf_upts_kern(self):
        normtcorf_fpts, divtconf_upts = self._scal_fpts[0], self._scal_upts_outb
        return self._be.kernel('mul', self._m3, normtcorf_fpts, divtconf_upts)

    def _gen_m0(self, dims, upts, ubasis, fpts):
        """Discontinuous soln at upts to discontinuous soln at fpts"""
        m = np.empty((len(fpts), len(upts)))

        ubasis_l = [lambdify(dims, u) for u in ubasis]
        for i,j in ndrange(*m.shape):
            m[i,j] = ubasis_l[j](*fpts[i])

        return m

    def _gen_m1(self, dims, upts, ubasis):
        """Trans discontinuous flux at upts to trans divergence of
        trans discontinuous flux at upts
        """
        m = np.empty((len(upts), len(dims), len(upts)))

        for j,k in ndrange(len(dims), len(upts)):
            p = lambdify(dims, ubasis[k].diff(dims[j]))
            for i in xrange(len(upts)):
                m[i,j,k] = p(*upts[i])

        return m.reshape(len(upts), -1)

    def _gen_m2(self, dims, ubasis, fpts, normfpts):
        """Trans discontinuous flux at upts to trans normal
        discontinuous flux at fpts
        """
        m = np.empty((len(fpts), len(dims), len(ubasis)))

        ubasis_l = [lambdify(dims, u) for u in ubasis]
        for i,k in ndrange(len(fpts), len(ubasis)):
            m[i,:,k] = normfpts[i,:]*ubasis_l[k](*fpts[i])

        return m.reshape(len(fpts), -1)

    def _gen_m3(self, dims, upts, fbasis):
        """Trans normal correction flux at upts to trans divergence of
        trans correction flux at upts
        """
        m = np.empty((len(upts), len(fbasis)))

        fbasis_l = [lambdify(dims, f) for f in fbasis]
        for i,j in ndrange(*m.shape):
            m[i,j] = fbasis_l[j](*upts[i])

        return m

    def _gen_djac_smat_upts(self, dims, eles, sbasis, upts):
        jacs = self._gen_jac_eles(dims, eles, sbasis, upts)
        smats, djacs = self._gen_smats(jacs, retdets=True)

        neles, ndims = eles.shape[1:]
        return djacs.reshape(-1, neles), smats.reshape(-1, ndims**2)

    def _gen_trans_fpts(self, dims, eles, sbasis, fpts, normfpts):
        jac = self._gen_jac_eles(dims, eles, sbasis, fpts)
        smats = self._gen_smats(jac)

        # Reshape (nfpts*neles, ndims, dims) => (nfpts, neles, ndims, ndims)
        smats = smats.reshape(len(fpts), -1, len(dims), len(dims))

        # We need to compute ||J|*[(J^{-1})^{T}.N]| where J is the
        # Jacobian and N is the normal for each fpt.  Using
        # J^{-1} = S/|J| where S are the smats, we have |S^{T}.N|.
        tnormfpts = np.einsum('ijlk,il->ijk', smats, normfpts)
        tnormfpts2 = np.einsum('...i,...i', tnormfpts, tnormfpts)

        return np.sqrt(tnormfpts2)

    def _gen_jac_eles(self, dims, eles, basis, pts):
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

    @abstractmethod
    def _gen_smats(self, jac, retdets=False):
        pass

    @abstractmethod
    def _gen_upts_basis(self, dims, order):
        pass

    @abstractmethod
    def _gen_fpts_basis(self, dims, order):
        pass

    @abstractmethod
    def _gen_spts_basis(self, dims, nspts):
        pass

    @abstractmethod
    def _gen_norm_fpts(self, dims, order):
        pass


class ElementsBase2d(ElementsBase):
    def __init__(self, be, eles, nsubanks, cfg):
        super(ElementsBase2d, self).__init__(be, eles, sy.symbols('p q'),
                                             nsubanks, cfg)

    def _gen_smats(self, jac, retdets=False):
        a, b, c, d = [jac[:,i,j] for i,j in ndrange(2,2)]

        smats = np.empty_like(jac)
        smats[:,0,0], smats[:,0,1] =  d, -b
        smats[:,1,0], smats[:,1,1] = -c,  a

        if retdets:
            return smats, a*c - b*d
        else:
            return smats


class ElementsBase3d(ElementsBase):
    def __init__(self, be, eles, nsubanks, cfg):
        super(ElementsBase3d, self).__init__(be, eles, sy.symbols('p q r'),
                                             nsubanks, cfg)

    def _gen_smats(self, jac, retdets=False):
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
