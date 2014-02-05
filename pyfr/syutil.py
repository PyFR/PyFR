# -*- coding: utf-8 -*-

import re

import sympy as sy
from sympy.mpmath import mp
from sympy.utilities.lambdify import lambdastr


def lambdify_mpf(dims, exprs):
    # Perform the initial lambdification
    ls = [lambdastr(dims, ex.evalf(mp.dps)) for ex in exprs]
    csf = {}

    # Locate all numerical constants in these lambdified expressions
    for l in ls:
        for m in re.findall(r'([0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?)', l):
            if m not in csf:
                csf[m] = mp.mpf(m)

    # Sort the keys by their length to prevent erroneous substitutions
    cs = sorted(csf, key=len, reverse=True)

    # Name these constants
    csn = {s: '__c%d' % i for i, s in enumerate(cs)}
    cnf = {n: csf[s] for s, n in csn.iteritems()}

    # Substitute
    lex = []
    for l in ls:
        for s in cs:
            l = l.replace(s, csn[s])
        lex.append(eval(l, cnf))

    return lex


def lambdify_jac_mpf(dims, exprs):
    jac_exprs = [ex.diff(d) for ex in exprs for d in dims]
    return lambdify_mpf(dims, jac_exprs)


def lagrange_basis(points, sym):
    """Generates a basis of polynomials, :math:`l_i(x)`, such that
    .. math::
       l_i(x) = \delta^x_{p_i}
    where :math:`p_i` is the i'th entry in *points* and :math:`x \in p`.
    """
    n = len(points)
    lagrange_poly = sy.interpolating_poly

    return [lagrange_poly(n, sym, points, [0]*i + [1] + [0]*(n-i-1)).expand()
            for i in xrange(n)]


def nodal_basis(points, basis, dims=None, basisfns=None):
    # If necessary lambdify the basis
    basisfns = basisfns or lambdify_mpf(dims, basis)

    # Evaluate the terms of the Vandermonde matrix
    V = [[fn(*p) for p in points] for fn in basisfns]

    # Invert this matrix to obtain the expansion coefficients
    Vinv = mp.matrix(V)**-1

    # Each nodal basis function is a linear combination of
    # orthonormal basis functions
    return [sum(c*b for c, b in zip(Vinv[i,:], basis))
            for i in xrange(len(points))]
