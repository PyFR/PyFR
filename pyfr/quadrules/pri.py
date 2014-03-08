# -*- coding: utf-8 -*-

from mpmath import mp

from pyfr.quadrules.base import BaseQuadRule
from pyfr.quadrules.line import BaseLineQuadRule
from pyfr.quadrules.tri import BaseTriQuadRule
from pyfr.util import subclass_map


class BasePriQuadRule(BaseQuadRule):
    eletype = 'pri'


class TensorProdPriQuadRule(BasePriQuadRule):
    def __init__(self, npts):
        roots = mp.polyroots([1, 1, 0, -2*npts])
        roots = [int(x) for x in roots if mp.isint(x) and x > 0]

        if not roots:
            raise ValueError('Invalid number of points for quad rule')

        tname, lname = self.name.split('*')

        trule_map = subclass_map(BaseTriQuadRule, 'name')
        trule = trule_map[tname](roots[0]*(roots[0] + 1) // 2)

        lrule_map = subclass_map(BaseLineQuadRule, 'name')
        lrule = lrule_map[lname](roots[0])

        self.points = [(t[0], t[1], l)
                       for l in lrule.points for t in trule.points]

        if hasattr(trule, 'weights') and hasattr(lrule, 'weights'):
            self.weights = [i*j for j in lrule.weights for i in trule.weights]


class WilliamsShunnGaussLegendrePriQuadRule(TensorProdPriQuadRule):
    name = 'williams-shunn*gauss-legendre'


class WilliamsShunnGaussLegendreLobattoPriQuadRule(TensorProdPriQuadRule):
    name = 'williams-shunn*gauss-legendre-lobatto'
