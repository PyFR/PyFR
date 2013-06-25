# -*- coding: utf-8 -*-

from abc import abstractmethod
import itertools as it
import os
import pkgutil
import re

from sympy.mpmath import mp


class BaseQuadRule(object):
    eletype = None
    orbits = None

    @abstractmethod
    def __init__(self):
        pass


class BaseAlgebraicQuadRule(BaseQuadRule):
    pass


class BaseTabulatedQuadRule(BaseQuadRule):
    def __init__(self, rule):
        pts, wts = [], []

        # Pre-process; here we remove white space, convert to upper-
        # case, ensure points and weights are delimited by newlines,
        # and remove any encasing brackets
        rule = re.sub(r'\s+', '', rule).upper()
        rule = re.sub(r'(?<=\)),?(?!$)', r'\n', rule)
        rule = rule[1:-1] if rule.startswith('[') else rule
        
        for s in rule.split('\n'):
            try:
                typ, orb, args = re.match(r'(PTS|WTS)(\d+)\((.*)\)$', s).groups()
            except TypeError:
                raise ValueError('Invalid quadrature rule syntax')

            # Convert the arguments to mp.mpf types
            args = map(mp.mpf, args.split(','))

            if typ == 'PTS':
                pts.append(self.orbits[orb](*args))
            else:
                wts.append([args[0]]*len(pts[-1]))

                if len(wts) != len(pts):
                    raise ValueError('Invalid weights')

        # Flattern
        self.points = list(it.chain.from_iterable(pts))
        self.weights = list(it.chain.from_iterable(wts))
            

class BaseStoredQuadRule(BaseTabulatedQuadRule):
    def __init__(self, npts):
        ptsname = '%s-%d.txt' % (self.name, npts)
        ptspath = os.path.join(self.eletype, ptsname)
        
        rule = pkgutil.get_data(__name__, ptspath)
        super(BaseStoredQuadRule, self).__init__(rule)
        