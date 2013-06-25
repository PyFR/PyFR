# -*- coding: utf-8 -*-

from abc import abstractproperty
import itertools as it

import numpy as np
import sympy as sy

from pyfr.bases.base import BasisBase
from pyfr.quadrules import BaseLineQuadRule, get_quadrule
from pyfr.syutil import lagrange_basis
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
    basis = [lagrange_basis(p, d) for d in reversed(dims)]

    # Take the cartesian product of these and multiply the resulting tuples
    cpbasis = np.array([np.prod(b) for b in it.product(*basis)])

    return cpbasis if compact else cpbasis.reshape((len(p),)*len(dims))


_quad_map_rots_np = np.array([[[ 1,  0], [ 0,  1]],
                              [[ 0,  1], [-1,  0]],
                              [[-1,  0], [ 0, -1]],
                              [[ 0, -1], [ 1,  0]]])


def quad_map_edge(fpts):
    mfpts = np.empty((4,) + fpts.shape, dtype=fpts.dtype)

    for i, frot in enumerate(_quad_map_rots_np):
        mfpts[i,...] = np.dot(fpts, frot)

    return mfpts


# Cube map face rotation scheme to go from face 1 -> 0..5
_cube_map_rots = np.array([
    [[-1,  0,  0], [ 0,  0,  1], [ 0,  1,  0]],   # 1 -> 0
    [[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]],   # 1 -> 1 (ident)
    [[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]],   # 1 -> 2
    [[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]],   # 1 -> 3
    [[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]],   # 1 -> 4
    [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]]])  # 1 -> 5


def cube_map_face(fpts):
    """Given a matrix of points (p,q,r) corresponding to face one of
    `the cube' this method maps these points onto the remaining faces

    On a cube parameterized by (p,q,r) -> (-1,-1,-1) × (1,1,1) face one
    is defined by (-1,-1,-1) × (1,-1,1)."""
    mfpts = np.empty((6,) + fpts.shape, dtype=fpts.dtype)

    for i, frot in enumerate(_cube_map_rots):
        mfpts[i,...] = np.dot(fpts, frot)

    return mfpts


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

        if self.nspts:
            # Root the number of shape points to get the # in each dim
            self._nsptsord = sy.S(self.nspts)**(sy.S(1)/self.ndims)

            if not self._nsptsord.is_Number:
                raise ValueError('Invalid number of shape points for {} dims'
                                 .format(self.ndims))

    @classmethod
    def std_ele(cls, sptord):
        esqr = get_quadrule(BaseLineQuadRule, 'equi-spaced', sptord + 1)
        return cart_prod_points(esqr.points, cls.ndims)

    @lazyprop
    def _pts1d(self):
        rule = self._cfg.get('mesh-elements-' + self.name, 'quad-rule')
        return get_quadrule(BaseLineQuadRule, rule, self._order + 1).points

    @lazyprop
    def upts(self):
        return cart_prod_points(self._pts1d, self.ndims)

    @lazyprop
    def ubasis(self):
        return nodal_basis(self._pts1d, self._dims)

    @lazyprop
    def spts1d(self):
        esqr = get_quadrule(BaseLineQuadRule, 'equi-spaced', self._nsptsord)
        return esqr.points

    @lazyprop
    def spts(self):
        return cart_prod_points(self.spts1d, self.ndims)

    @lazyprop
    def sbasis(self):
        return nodal_basis(self.spts1d, self._dims)

    @property
    def nupts(self):
        return (self._order + 1)**self.ndims

    @property
    def nfacefpts(self):
        return [(self._order + 1)**(self.ndims - 1)] * (2*self.ndims)

    @abstractproperty
    def _rschemes(self):
        pass

    def fpts_idx_for_face(self, face, rtag):
        return self._rschemes[face, rtag]


class QuadBasis(TensorProdBasis, BasisBase):
    name = 'quad'
    ndims = 2

    @lazyprop
    def fpts(self):
        # Get the 1D points
        pts1d = self._pts1d

        # Edge zero has points (q,-1)
        ezeropts = np.empty((len(pts1d), 2), dtype=np.object)
        ezeropts[:,0] = pts1d
        ezeropts[:,1] = -1

        # Quad map edge zero to get the full set
        return quad_map_edge(ezeropts).reshape(-1, 2)

    @lazyprop
    def fbasis(self):
        # Get the 1D points
        pts1d = self._pts1d

        # Allocate space for the flux basis
        fbasis = np.empty((4, len(pts1d)), dtype=np.object)

        # Pair up opposite edges with their associated (normal) dimension
        for epair, sym in zip([(3,1), (0,2)], self._dims):
            nbdim = [d for d in self._dims if d is not sym]
            fbasis[epair,...] = nodal_basis(pts1d, nbdim, compact=False)

            eta = self._cfg.get('mesh-elements-quad', 'vcjh-eta')
            diffcorfn = diff_vcjh_correctionfn(self._order, eta, sym)

            for p, gfn in zip(epair, diffcorfn):
                if p in (2,3):
                    fbasis[p] = fbasis[p,::-1]
                fbasis[p,:] *= gfn

        # Correct faces with negative normals
        fbasis[(3,0),:] *= -1

        return fbasis.ravel()

    @lazyprop
    def norm_fpts(self):
        # Normals for edge zero are (0,-1)
        ezeronorms = np.zeros((self._order + 1, 2), dtype=np.int)
        ezeronorms[:,1] = -1

        # Edge map
        return quad_map_edge(ezeronorms).reshape(-1, 2)

    @lazyprop
    def _rschemes(self):
        k = self._order + 1

        # Pre-compute all possible flux point rotation schemes
        rs = np.empty((4, 2), dtype=np.object)
        for face, rtag in ndrange(*rs.shape):
            fpts = np.arange(face*k, (face + 1)*k)

            if rtag == 0:
                pass
            elif rtag == 1:
                fpts = fpts[::-1]

            rs[face,rtag] = fpts

        return rs


class HexBasis(TensorProdBasis, BasisBase):
    name = 'hex'
    ndims = 3

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
        for fpair, sym in zip([(4,2), (1,3), (0,5)], self._dims):
            nbdims = [d for d in self._dims if d is not sym]
            fbasis[fpair,...] = nodal_basis(pts1d, nbdims, compact=False)

            eta = self._cfg.get('mesh-elements-hex', 'vcjh-eta')
            diffcorfn = diff_vcjh_correctionfn(self._order, eta, sym)

            for p, gfn in zip(fpair, diffcorfn):
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

    @lazyprop
    def _rschemes(self):
        k = self._order + 1

        # Compute all possible flux point rotation schemes
        rs = np.empty((6, 5), dtype=np.object)
        for face, rtag in ndrange(*rs.shape):
            fpts = np.arange(face*k*k, (face + 1)*k*k).reshape(k,k)

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

        return rs
