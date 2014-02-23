# -*- coding: utf-8 -*-

from sympy.mpmath import mp

from pyfr.quadrules.base import BaseQuadRule, BaseStoredQuadRule, perm_orbit


class BaseTetQuadRule(BaseQuadRule):
    eletype = 'tet'
    orbits = {
        '4': perm_orbit('a', ['a', 'a', 'a', 'a']),
        '31': perm_orbit('a', ['a', 'a', 'a', '1 - 3*a']),
        '22': perm_orbit('a', ['a', 'a', '0.5 - a', '0.5 - a']),
        '211': perm_orbit('ab', ['a', 'a', 'b', '1 - 2*a - b']),
        '1111': perm_orbit('abc', ['a', 'b', 'c', '1 - a - b - c'])
    }

    def __init__(self, npts):
        super(BaseTetQuadRule, self).__init__(npts)

        # Convert from barycentric to Cartesian
        xyz = [(-1, 1, -1, -1), (-1, -1, 1, -1), (-1, -1, -1, 1)]
        self.points = [[mp.fdot(p, t) for t in xyz] for p in self.points]

        # Scale the weights by |J| = 4/3
        self.weights = [3*w / 4 for w in self.weights]


class ShunnHamTetQuadRule(BaseTetQuadRule, BaseStoredQuadRule):
    name = 'shunn-ham'
