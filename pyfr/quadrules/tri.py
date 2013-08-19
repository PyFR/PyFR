# -*- coding: utf-8 -*-

from pyfr.quadrules.base import BaseQuadRule, BaseStoredQuadRule


class BaseTriQuadRule(BaseQuadRule):
    eletype = 'tri'
    orbits = {
        '3': lambda a: [(a, a, a)],
        '21': lambda a: [(a, a, 1-2*a), (a, 1-2*a, a), (1-2*a, a, a)],
        '111': lambda a, b: [(a, b, 1-a-b), (a, 1-a-b, b), (b, a, 1-a-b),
                             (b, 1-a-b, a), (1-a-b, a, b), (1-a-b, b, a)]}


class AlphaOptTriQuadRule(BaseStoredQuadRule, BaseTriQuadRule):
    name = 'alpha-opt'


class WilliamsShunnTriQuadRule(BaseStoredQuadRule, BaseTriQuadRule):
    name = 'williams-shunn'
