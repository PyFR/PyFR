# -*- coding: utf-8 -*-

from pyfr.quadrules.base import BaseQuadRule, BaseStoredQuadRule, perm_orbit


class BaseTriQuadRule(BaseQuadRule):
    eletype = 'tri'
    orbits = {
        '3': perm_orbit('a', ['a', 'a', 'a']),
        '21': perm_orbit('a', ['a', 'a', '1 - 2*a']),
        '111': perm_orbit('ab', ['a', 'b', '1 - a - b'])}


class AlphaOptTriQuadRule(BaseTriQuadRule, BaseStoredQuadRule):
    name = 'alpha-opt'


class WilliamsShunnTriQuadRule(BaseTriQuadRule, BaseStoredQuadRule):
    name = 'williams-shunn'
