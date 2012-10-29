# -*- coding: utf-8 -*-

from pyfr.elements.base import ElementsBase3d

from pyfr.polys import (gauss_legendre_points,
                        gauss_legendre_lobatto_points)

import numpy as np
import sympy as sy

import itertools

def cart_prod_points(points, ndim, compact=True):
    """Performs a cartesian product extension of *points* into *ndim*

    For idiosyncratic reason the counting order of indices is from
    first to last, i.e, it is the first index the counts quickest,
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

    cprodpts = np.empty((npoints,)*ndim + (ndim,))
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

    return [sy.interpolating_poly(n, sym, points, (0,)*i + (1,) + (0,)*(n-i))
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
_cube_map_rots_np = [np.asanyarray(sy.matrix2numpy(r), dtype=np.float64)
                     for r in _cube_map_rots_sy]

def cube_map_face(fpoints):
    """Given a matrix of points (p,q,r) corresponding to face one of
    `the cube' this method maps these points onto the remaining faces

    On a cube parameterized by (p,q,r) -> (-1,-1,-1) × (1,1,1) face one
    is defined by (-1,-1,-1) × (1,-1,1)."""
    mfpoints = np.empty((6,) + fpoints.shape)

    for i,frot in enumerate(_cube_map_rots_np):
        mfpoints[i,...] = np.dot(fpoints, frot)

    return mfpoints

def diff_vcjh_correctionfn(k, c, sym):
    # Expand shorthand forms of c for common schemes
    ccommon = dict(dg='0',
                   sd='2*k/((2*k+1)*(k+1)*(ak*k!)^2)',
                   hu='2*(k+1)/(k*(2*k+1)*(ak*k!)^2)')

    ak = sy.binomial(2*k, k)/2**k
    c = sy.S(ccommon.get(c, c), locals=dict(k=k, ak=ak))
    eta_k = c*(2*k + 1)*(ak*sy.factorial(k))**2/2

    lkm1, lk, lkp1 = [sy.legendre_poly(m, sym) for m in [k-1, k, k+1]]

    # Correction function derivatives, Eq. 3.46 and 3.47
    diffgr = (sy.S(1)/2 * (lk + (eta_k*lkm1 + lkp1)/(1 + eta_k))).diff()
    diffgl = -diffgr.subs(sym, -sym)

    return diffgl, diffgr

class TensorProdBase(object):
    def _gen_upts_basis(self, dims, order):
        pts1d = gauss_legendre_points(order+1)

        pts = cart_prod_points(pts1d, 3)
        basis = nodal_basis(pts1d, dims)

        return pts, basis

    def _gen_spts_basis(self, dims, nspts):
        # Root the number of shape points to get the # in each dim
        nsptsord = sy.S(nspts)**(sy.S(1)/len(dims))

        if not nsptsord.is_Number:
            raise ValueError('Invalid number of shape points for {} dims'\
                             .format(len(dims)))

        # Generate nsptsord equispaced points from (-1, 1)
        pts1d = np.linspace(-1, 1, nsptsord+2)[1:-1]

        pts = cart_prod_points(pts1d, len(dims))
        basis = nodal_basis(pts1d, dims)

        return pts, basis


class Hexahedra(TensorProdBase, ElementsBase3d):
    def _gen_fpts_basis(self, dims, order):
        # Get the 1D points
        pts1d = gauss_legendre_points(order+1)

        # Perform a 2D extension to get the (p,r) points of face one
        pts2d = cart_prod_points(pts1d, 2, compact=False)

        # 3D points are just (p,-1,r) for face one
        fonepts = np.empty(pts2d.shape[:-1] + (3,))
        fonepts[...,(0,2)] = pts2d
        fonepts[...,1] = -1

        # Cube map face one to get faces zero through five
        fpts = cube_map_face(fonepts)

        # Allocate space for the flux points basis
        fbasis = np.empty(fpts.shape[:-1], dtype=np.object)

        # Pair up opposite faces with their associated (normal) dimension
        for fpair,sym in zip([(4,2), (1,3), (0,5)], dims):
            nbdims = [d for d in dims if d is not sym]
            fbasis[fpair,...] = nodal_basis(pts1d, nbdims, compact=False)


            c = self._cfg.get('scheme', 'c')
            diffcorfn = diff_vcjh_correctionfn(order, c, sym)

            for p,gfn in zip(fpair, diffcorfn):
                if p in (0,3,4):
                    fbasis[p] = np.fliplr(fbasis[p])
                fbasis[p,...] *= gfn

        # Correct faces with negative normals
        fbasis[(4,1,0),...] *= -1

        return fpts.reshape(-1,3), fbasis.reshape(-1)

    def _gen_norm_fpts(self, dims, order):
        # Normals for face one are (0,-1,0)
        fonenorms = np.zeros((order+1,)*2 + (3,))
        fonenorms[...,1] = -1

        # Cube map to get the remaining face normals
        return cube_map_face(fonenorms).reshape(-1, 3)
