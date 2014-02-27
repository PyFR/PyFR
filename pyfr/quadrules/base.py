# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import itertools as it
import os
import pkgutil
import re

from sympy.mpmath import mp
import numpy as np

from pyfr.util import lazyprop


def perm_orbit(args, coords):
    # Argument list for the function
    args = ', '.join(args)

    # Compute all unique permutations of coords
    perms = set(it.permutations(coords))

    # Suitably format
    body = '[' + ', '.join('(' + ', '.join(p) + ')' for p in perms) + ']'

    return eval('lambda {}: {}'.format(args, body))


class BaseQuadRule(object):
    __metaclass__ = ABCMeta

    eletype = None
    orbits = None

    @abstractmethod
    def __init__(self):
        pass

    @lazyprop
    def np_points(self):
        return np.asanyarray(self.points, dtype=np.float)

    @lazyprop
    def np_weights(self):
        return np.asanyarray(self.weights, dtype=np.float)


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
                m = re.match(r'(PTS|WTS)(\d+)\((.*)\)$', s)
                typ, orb, args = m.groups()
            except TypeError:
                raise ValueError('Invalid quadrature rule syntax')

            # Convert the arguments to mp.mpf types
            args = [mp.mpf(arg) for arg in args.split(',')]

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
