# -*- coding: utf-8 -*-

import sympy as sy


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


def norm_jacobi(n, a, b, sym):
    G, F = sy.gamma, sy.factorial

    N2 = sy.S(2)**(a + b + 1)/(2*n + a + b + 1)\
       * (G(n + a + 1)*G(n + b + 1))/(F(n)*G(n + a + b + 1))

    return sy.jacobi(n, a, b, sym) / sy.sqrt(N2)
