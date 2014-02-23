# -*- coding: utf-8 -*-

from sympy.mpmath import mp

from pyfr.quadrules.base import BaseQuadRule, BaseAlgebraicQuadRule


class BaseLineQuadRule(BaseQuadRule):
    eletype = 'line'
    orbits = {
        '2': lambda a: [a],
        '11': lambda a: [-a, a]
    }


class GaussLegendreQuadRule(BaseLineQuadRule, BaseAlgebraicQuadRule):
    name = 'gauss-legendre'

    def __init__(self, npts):
        # Legendre poly
        lp = lambda x: mp.legendre(npts, x)

        self.points = mp.polyroots(mp.taylor(lp, 0, npts)[::-1])
        self.weights = [2/((1 - p*p)*mp.diff(lp, p)**2) for p in self.points]


class GaussLegendreLobattoQuadRule(BaseLineQuadRule, BaseAlgebraicQuadRule):
    name = 'gauss-legendre-lobatto'

    def __init__(self, npts):
        # Legendre poly
        lp = lambda x: mp.legendre(npts - 1, x)

        # Coefficients of lp
        cf = mp.taylor(lp, 0, npts - 1)

        # Coefficients of dlp/dx
        dcf = [i*c for i, c in enumerate(cf[1:], start=1)]

        self.points = [mp.mpf(-1)] + mp.polyroots(dcf[::-1]) + [mp.mpf(1)]
        self.weights = [2/(npts*(npts - 1)*lp(p)**2) for p in self.points]
        

class GaussChebyshevQuadRule(BaseLineQuadRule, BaseAlgebraicQuadRule):
    name = 'gauss-chebyshev'
    
    def __init__(self, npts):
        # Only points
        self.points = [mp.cos((2*i - 1)*mp.pi/(2*npts))
                       for i in xrange(npts, 0, -1)]
                       

class GaussChebyshevLobattoQuadRule(BaseLineQuadRule, BaseAlgebraicQuadRule):
    name = 'gauss-chebyshev-lobatto'
    
    def __init__(self, npts):
        # Only points
        self.points = [mp.cos((i - 1)*mp.pi/(npts - 1))
                       for i in xrange(npts, 0, -1)]
 
 
class EquiSpacedQuadRule(BaseLineQuadRule, BaseAlgebraicQuadRule):
    name = 'equi-spaced'
    
    def __init__(self, npts):
        # Only points
        self.points = mp.linspace(-1, 1, npts)
