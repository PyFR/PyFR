# -*- coding: utf-8 -*-

import itertools

import numpy as np
import sympy as sy

from pyfr.bases.base import BasisBase
from pyfr.quad_points import points_for_rule, equi_spaced
from pyfr.util import ndrange, lazyprop


def cart_prod_points(points, ndim, compact=True):
    """Performs a cartesian product extension of *points* into *ndim*

    For idiosyncratic reason the counting order of indices is from
    first to last, i.e, it is the first index that counts quickest,
    followed by the second index and so on.

    **Example**
    >>> cart_prod_points([-1, 0, 1], 2)
    array([[-1., -1.],
       [ 0., -1.],
       [ 1., -1.],
       [-1.,  0.],
       [ 0.,  0.],
       [ 1.,  0.],
       [-1.,  1.],
       [ 0.,  1.],
       [ 1.,  1.]])
    """
    npoints = len(points)

    cprodpts = np.empty((npoints,)*ndim + (ndim,), dtype=np.object)
    for i,ax in enumerate(np.ix_(*(points,)*ndim)):
        # -i-1 ensures we count first-to-last
        cprodpts[...,-i-1] = ax

    # Compact into an array of ndim component tuples
    if compact:
        return cprodpts.reshape(-1, ndim)
    else:
        return cprodpts

def _nodal_basis1d(points, sym):
    """Generates a basis of polynomials, :math:`l_i(x)`, such that
    .. math::
       l_i(x) = \delta^x_{p_i}
    where :math:`p_i` is the i'th entry in *points* and :math:`x \in p`.
    """
    n = len(points)
    lagrange_poly = sy.interpolating_poly

    return [lagrange_poly(n, sym, points, (0,)*i + (1,) + (0,)*(n-i)).expand()
            for i in xrange(n)]

def nodal_basis(points, dims, compact=True):
    """Generates a nodal basis for *points* over *dims*

    .. note::
      This function adcfg the same first-to-last counting order as
      :func:`cart_prod_points` with the first index varying quickest.

    **Example**
    >>> import sympy as sy
    >>> nb = nodal_basis([-1, 1], sy.symbols('p q'))
    >>> nb[0]
    (-p/2 + 1/2)*(-q/2 + 1/2)
    >>> nb[0].subs(dict(p=-1, q=-1))
    1
    >>> nb[0].subs(dict(p=1, q=-1))
    0
    """
    p = list(points)

    # Evaluate the basis function in terms of each dimension
    basis = [_nodal_basis1d(p, d) for d in reversed(dims)]

    # Take the cartesian product of these and multiply the resulting tuples
    cpbasis = np.array([np.prod(b) for b in itertools.product(*basis)])

    return cpbasis if compact else cpbasis.reshape((len(p),)*len(dims))

# Cube map face rotation scheme to go from face 1 -> 0..5
_cube_map_rots_sy = [sy.rot_axis2(-sy.pi)*sy.rot_axis1(sy.pi/2),
                     sy.eye(3),
                     sy.rot_axis3(sy.pi/2),
                     sy.rot_axis3(sy.pi),
                     sy.rot_axis3(-sy.pi/2),
                     sy.rot_axis1(-sy.pi/2)]

# Rotation scheme as numpy arrays
_cube_map_rots_np = [np.asanyarray(sy.matrix2numpy(r), dtype=np.float)
                     for r in _cube_map_rots_sy]

def cube_map_face(fpoints):
    """Given a matrix of points (p,q,r) corresponding to face one of
    `the cube' this method maps these points onto the remaining faces

    On a cube parameterized by (p,q,r) -> (-1,-1,-1) × (1,1,1) face one
    is defined by (-1,-1,-1) × (1,-1,1)."""
    mfpoints = np.empty((6,) + fpoints.shape, dtype=fpoints.dtype)

    for i,frot in enumerate(_cube_map_rots_np):
        mfpoints[i,...] = np.dot(fpoints, frot)

    return mfpoints

def diff_vcjh_correctionfn(k, eta, sym):
    # Expand shorthand forms of eta_k for common schemes
    etacommon = dict(dg='0', sd='k/(k+1)', hu='(k+1)/k')

    eta_k = sy.S(etacommon.get(eta, eta), locals=dict(k=k))

    lkm1, lk, lkp1 = [sy.legendre_poly(m, sym) for m in [k-1, k, k+1]]

    # Correction function derivatives, Eq. 3.46 and 3.47
    diffgr = (sy.S(1)/2 * (lk + (eta_k*lkm1 + lkp1)/(1 + eta_k))).diff()
    diffgl = -diffgr.subs(sym, -sym)

    return diffgl, diffgr

class TensorProdBasis(object):
    def __init__(self, *args, **kwargs):
        super(TensorProdBasis, self).__init__(*args, **kwargs)

        # Root the number of shape points to get the # in each dim
        self._nsptsord = sy.S(self.nspts)**(sy.S(1)/self.ndims)

        if not self._nsptsord.is_Number:
            raise ValueError('Invalid number of shape points for {} dims'\
                             .format(self.ndims))

    @lazyprop
    def _pts1d(self):
        rule = self._cfg.get('mesh-elements', 'quad-rule')
        return points_for_rule(rule, self._order + 1)

    @lazyprop
    def upts(self):
        return cart_prod_points(self._pts1d, self.ndims)

    @lazyprop
    def ubasis(self):
        return nodal_basis(self._pts1d, self._dims)

    @lazyprop
    def spts(self):
        return cart_prod_points(equi_spaced(self._nsptsord), self.ndims)

    @lazyprop
    def sbasis(self):
        return nodal_basis(equi_spaced(self._nsptsord), self._dims)

    @property
    def nupts(self):
        return (self._order + 1)**self.ndims

    @property
    def nfpts(self):
        return [(self._order + 1)**(self.ndims - 1)] * (2*self.ndims)


class HexBasis(TensorProdBasis, BasisBase):
    name = 'hex'
    ndims = 3

    def __init__(self, *args, **kwargs):
        super(HexBasis, self).__init__(*args, **kwargs)

        k = self._order + 1

        # Pre-compute all possible flux point rotation schemes
        self._rschemes = rs = np.empty((6, 5), dtype=np.object)
        for face,rtag in ndrange(*rs.shape):
            fpts = np.arange(face*k*k, (face+1)*k*k).reshape(k,k)

            if rtag == 0:
                pass
            elif rtag == 1:
                fpts = np.fliplr(fpts)
            elif rtag == 2:
                fpts = np.fliplr(fpts)[::-1]
            elif rtag == 3:
                fpts = np.transpose(fpts)
            elif rtag == 4:
                fpts = np.transpose(fpts)[::-1]

            rs[face,rtag] = fpts.ravel()

    @lazyprop
    def fpts(self):
        # Get the 1D points
        pts1d = self._pts1d

        # Perform a 2D extension to get the (p,r) points of face one
        pts2d = cart_prod_points(pts1d, 2, compact=False)

        # 3D points are just (p,-1,r) for face one
        fonepts = np.empty(pts2d.shape[:-1] + (3,), dtype=np.object)
        fonepts[...,(0,2)] = pts2d
        fonepts[...,1] = -1

        # Cube map face one to get faces zero through five
        return cube_map_face(fonepts).reshape(-1, 3)

    @lazyprop
    def fbasis(self):
        # Get the 1D points
        pts1d = self._pts1d

        # Allocate space for the flux points basis
        fbasis = np.empty([6] + [self._order + 1]*2, dtype=np.object)

        # Pair up opposite faces with their associated (normal) dimension
        for fpair,sym in zip([(4,2), (1,3), (0,5)], self._dims):
            nbdims = [d for d in self._dims if d is not sym]
            fbasis[fpair,...] = nodal_basis(pts1d, nbdims, compact=False)

            eta = self._cfg.get('mesh-elements', 'vcjh-eta')
            diffcorfn = diff_vcjh_correctionfn(self._order, eta, sym)

            for p,gfn in zip(fpair, diffcorfn):
                if p in (0,3,4):
                    fbasis[p] = np.fliplr(fbasis[p])
                fbasis[p,...] *= gfn

        # Correct faces with negative normals
        fbasis[(4,1,0),...] *= -1

        return fbasis.ravel()

    @lazyprop
    def norm_fpts(self):
        # Normals for face one are (0,-1,0)
        fonenorms = np.zeros([self._order + 1]*2 + [3], dtype=np.int)
        fonenorms[...,1] = -1

        # Cube map to get the remaining face normals
        return cube_map_face(fonenorms).reshape(-1, 3)

    def fpts_idx_for_face(self, face, rtag):
        return self._rschemes[face, rtag]
