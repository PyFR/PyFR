# -*- coding: utf-8 -*-

import sympy as sy
from sympy.abc import x
from sympy.mpmath import mp

from pyfr.quadrules.base import BaseQuadRule, BaseAlgebraicQuadRule


class BaseLineQuadRule(BaseQuadRule):
    eletype = 'line'
    orbits = {'2': lambda a: [a],
              '11': lambda a: [-a, a]}


class GaussLegendreQuadRule(BaseAlgebraicQuadRule, BaseLineQuadRule):
    name = 'gauss-legendre'

    def __init__(self, npts):
        # Form a suitable Legendre poly
        Pn = sy.legendre_poly(npts, x)
        dPn = Pn.diff()
        
        # Roots
        self.points = mp.polyroots(map(mp.mpf, sy.Poly(Pn).all_coeffs()))
                
        # Weights
        self.weights = [2/((1 - p**2)*dPn.evalf(mp.dps, subs={x: p})**2)
                        for p in self.points]


class GaussLegendreLobattoQuadRule(BaseAlgebraicQuadRule, BaseLineQuadRule):
    name = 'gauss-legendre-lobatto'

    def __init__(self, npts):
        # Form a suitable Legendre poly
        Pn = sy.legendre_poly(npts - 1, x)
        dPn = Pn.diff()

        # Roots
        roots = mp.polyroots(map(mp.mpf, sy.Poly(dPn).all_coeffs()))
        self.points = [mp.mpf(-1)] + roots + [mp.mpf(1)]

        # Weights
        wts0 = mp.mpf(2)/(npts*(npts - 1))
        wtsi = [2/(npts*(npts - 1)*Pn.evalf(mp.dps, subs={x: p})**2)
                for p in self.points[1:-1]]
        self.weights = [wts0] + wtsi + [wts0]
        

class GaussChebyshevQuadRule(BaseAlgebraicQuadRule, BaseLineQuadRule):
    name = 'gauss-chebyshev'
    
    def __init__(self, npts):
        # Only points
        self.points = [mp.cos((2*i - 1)*mp.pi/(2*npts))
                       for i in xrange(npts, 0, -1)]
                       

class GaussChebyshevLobattoQuadRule(BaseAlgebraicQuadRule, BaseLineQuadRule):
    name = 'gauss-chebyshev-lobatto'
    
    def __init__(self, npts):
        # Only points
        self.points = [mp.cos((i - 1)*mp.pi/(npts - 1))
                       for i in xrange(npts, 0, -1)]
 
 
class EquiSpacedQuadRule(BaseLineQuadRule):
    name = 'equi-spaced'
    
    def __init__(self, npts):
        # Only points
        self.points = [mp.mpf(-1) + mp.mpf(2*i)/(npts - 1)
                       for i in xrange(npts)]
