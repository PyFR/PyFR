# -*- coding: utf-8 -*-

from sympy import Poly, legendre_poly, diff
from sympy.abc import x

from mpmath import polyroots

def gauss_legendre_points(n):
    """Returns the Gauss-Legendre quadrature points for order *n*

    These are defined as the roots of P_n where P_n is the n'th
    Legendre polynomial.
    """
    coeffs = Poly(legendre_poly(n, x)).all_coeffs()

    roots = polyroots([float(c) for c in coeffs])
    return [float(r) for r in roots]

def lobatto_points(n):
    """Returns the Lobatto quadrature points for order *n*

    These are defined as the roots of P'_(n-1) where P'_(n-1) is the
    first derivative of the n'th - 1 Legendre polynomial plus the
    points -1.0 and +1.0.
    """
    coeffs = Poly(diff(legendre_poly(n-1, x), x)).all_coeffs()

    roots = polyroots([float(c) for c in coeffs])
    return [-1.0] + [float(r) for r in roots] + [1.0]

def sympeval(poly, symbols, vals):
    """Evaluates a SymPy polynomial"""
    return float(poly.eval(dict(zip(dims, vals))))
