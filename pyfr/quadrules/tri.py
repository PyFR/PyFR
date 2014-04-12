# -*- coding: utf-8 -*-

from mpmath import mp

from pyfr.quadrules.base import BaseQuadRule, BaseStoredQuadRule, perm_orbit


class BaseTriQuadRule(BaseQuadRule):
    eletype = 'tri'
    orbits = {
        '3': perm_orbit('a', ['a', 'a', 'a']),
        '21': perm_orbit('a', ['a', 'a', '1 - 2*a']),
        '111': perm_orbit('ab', ['a', 'b', '1 - a - b'])
    }

    def __init__(self, npts):
        super(BaseTriQuadRule, self).__init__(npts)

        # Convert from barycentric to Cartesian
        xy = [(-1, 1, -1), (-1, -1, 1)]
        self.points = [[mp.fdot(p, t) for t in xy] for p in self.points]

        # Scale the weights by |J| = 2
        self.weights = [2*w for w in self.weights]


class AlphaOptTriQuadRule(BaseTriQuadRule, BaseStoredQuadRule):
    name = 'alpha-opt'


class WilliamsShunnTriQuadRule(BaseTriQuadRule, BaseStoredQuadRule):
    name = 'williams-shunn'
