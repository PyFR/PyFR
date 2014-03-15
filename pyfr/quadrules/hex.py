# -*- coding: utf-8 -*-

from mpmath import mp

from pyfr.quadrules.base import BaseQuadRule
from pyfr.quadrules.line import BaseLineQuadRule
from pyfr.util import subclass_where


class BaseHexQuadRule(BaseQuadRule):
    eletype = 'hex'


class TensorProdHexQuadRule(BaseHexQuadRule):
    def __init__(self, npts):
        if not mp.isint(mp.cbrt(npts)):
            raise ValueError('Invalid number of points for quad rule')

        rulecls = subclass_where(BaseLineQuadRule, name=self.name)
        rule = rulecls(int(mp.cbrt(npts)))

        pts = rule.points
        self.points = [(i, j, k) for k in pts for j in pts for i in pts]

        if hasattr(rule, 'weights'):
            wts = rule.weights
            self.weights = [i*j*k for k in wts for j in wts for i in wts]


class GaussLegendreHexQuadRule(TensorProdHexQuadRule):
    name = 'gauss-legendre'


class GaussLegendreLobattoHexQuadRule(TensorProdHexQuadRule):
    name = 'gauss-legendre-lobatto'


class GaussChebyshevHexQuadRule(TensorProdHexQuadRule):
    name = 'gauss-chebyshev'


class GaussChebyshevLobattoHexQuadRule(TensorProdHexQuadRule):
    name = 'gauss-chebyshev-lobatto'


class EquiSpacedHexQuadRule(TensorProdHexQuadRule):
    name = 'equi-spaced'
