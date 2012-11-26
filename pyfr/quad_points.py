# -*- coding: utf-8 -*-

import sympy as sy
import numpy as np

try:
    import mpmath as mp
except ImportError:
    import sympy.mpmath as mp

def points_for_rule(name, n):
    rules = {'equi-spaced': equi_spaced,
             'gauss-legendre': gauss_legendre,
             'gauss-legendre-lobatto': gauss_legendre_lobatto,
             'gauss-chebyshev': gauss_chebyshev,
             'gauss-chebyshev-lobatto': gauss_chebyshev_lobatto}
    return rules[name](n)

def equi_spaced(n):
    return list(np.linspace(-1.0, 1.0, n))

def gauss_legendre(n):
    """Returns the Gauss-Legendre quadrature points for order *n*

    These are defined as the roots of P_n where P_n is the n'th
    Legendre polynomial.
    """
    coeffs = sy.Poly(sy.legendre_poly(n)).all_coeffs()

    roots = mp.polyroots([float(c) for c in coeffs])
    return [float(r) for r in roots]

def gauss_legendre_lobatto(n):
    """Returns the Gauss-Legendre-Lobatto quadrature points for order *n*

    These are defined as the roots of P'_(n-1) where P'_(n-1) is the
    first derivative of the n'th - 1 Legendre polynomial plus the
    points -1.0 and +1.0.
    """
    coeffs = sy.Poly(sy.legendre_poly(n-1).diff()).all_coeffs()

    roots = mp.polyroots([float(c) for c in coeffs])
    return [-1.0] + [float(r) for r in roots] + [1.0]

def gauss_chebyshev(n):
    """Returns the Gauss-Chebyshev quadrature points for order *n*

    These are given by cos((2i - 1)/(2n) * pi) for i = 1..n
    """
    return [float(sy.cos((2*i - 1)*sy.pi/(2*n))) for i in xrange(n, 0, -1)]

def gauss_chebyshev_lobatto(n):
    """Returns the Gauss-Chebyshev-Lobatto quadrature points for order *n*

    These are given by cos(i*pi/n) for i = 1..n
    """
    return [float(sy.cos((i-1)*sy.pi/(n-1))) for i in xrange(n, 0, -1)]
