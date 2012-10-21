# -*- coding: utf-8 -*-

import sympy as sy
import numpy as np

import itertools

def lagrange_basis(points, sym):
    """Generates a basis of polynomials, :math:`l_i(x)`, such that
    .. math::
       l_i(x) =
       \begin{cases}
        0 & x\neq p_{i}\\
        1 & x=p_{i}
       \end{cases}
    where :math:`p_i` is the i'th entry in *points*
    """
    n = len(points)

    return [sy.interpolating_poly(n, sym, points, (0,)*i + (1,) + (0,)*(n-i))
            for i in xrange(n)]

def cart_prod_points(points, ndim):
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
    return cprodpts.reshape((np.prod(cprodpts.shape[:-1]), ndim))

def cart_prod_basis(points, dims, basisfn):
    """Performs a cartesian product extension of a basis

    .. note::
      This function adopts the same first-to-last counting order as
      :func:`cart_prod_points` with the first index varying quickest.

    **Example**
    >>> import sympy as sy
    >>> cb = cart_prod_basis([-1, 1], sy.symbols('p q'), lagrange_basis)
    >>> cb[0]
    (-p/2 + 1/2)*(-q/2 + 1/2)
    >>> cb[0].subs(dict(p=-1, q=-1))
    1
    >>> cb[0].subs(dict(p=1, q=-1))
    0
    """
    # Evaluate the basis function in terms of each dimension (r,q,p)
    basis = [basisfn(points, d) for d in reversed(dims)]

    # Take the cartesian product of these and multiply the resulting tuples
    return [np.prod(b) for b in itertools.product(*basis)]
