# -*- coding: utf-8 -*-

from mpmath import mp

from pyfr.quadrules.base import BaseQuadRule
from pyfr.quadrules.line import BaseLineQuadRule
from pyfr.util import subclass_map


class BaseQuadQuadRule(BaseQuadRule):
    eletype = 'quad'


class TensorProdQuadQuadRule(BaseQuadQuadRule):
    def __init__(self, npts):
        if not mp.isint(mp.sqrt(npts)):
            raise ValueError('Invalid number of points for quad rule')

        rule_map = subclass_map(BaseLineQuadRule, 'name')
        rule = rule_map[self.name](int(mp.sqrt(npts)))

        self.points = [(i, j) for j in rule.points for i in rule.points]

        if hasattr(rule, 'weights'):
            self.weights = [i*j for j in rule.weights for i in rule.weights]


class GaussLegendreQuadQuadRule(TensorProdQuadQuadRule):
    name = 'gauss-legendre'


class GaussLegendreLobattoQuadQuadRule(TensorProdQuadQuadRule):
    name = 'gauss-legendre-lobatto'


class GaussChebyshevQuadQuadRule(TensorProdQuadQuadRule):
    name = 'gauss-chebyshev'


class GaussChebyshevLobattoQuadQuadRule(TensorProdQuadQuadRule):
    name = 'gauss-chebyshev-lobatto'


class EquiSpacedQuadQuadRule(TensorProdQuadQuadRule):
    name = 'equi-spaced'
