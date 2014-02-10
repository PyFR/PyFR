# -*- coding: utf-8 -*-

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


class ShunnHamTetQuadRule(BaseStoredQuadRule, BaseTetQuadRule):
    name = 'shunn-ham'
