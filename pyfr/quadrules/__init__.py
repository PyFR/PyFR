# -*- coding: utf-8 -*-

from pkg_resources import resource_listdir, resource_string
import re

from mpmath import mp
import numpy as np

from pyfr.util import lazyprop


class BaseTabulatedQuadRule(object):
    def __init__(self, rule):
        self.points = []
        self.weights = []

        rule = re.sub(r'(?<=\))\s*,?\s*(?!$)', r'\n', rule)
        rule = re.sub(r'\(|\)|,', '', rule).strip()
        rule = rule[1:-1] if rule.startswith('[') else rule

        for l in rule.splitlines():
            if not l:
                continue

            # Parse the line
            args = [mp.mpf(f) for f in l.split()]

            if len(args) == self.ndim:
                self.points.append(args)
            elif len(args) == self.ndim + 1:
                self.points.append(args[:-1])
                self.weights.append(args[-1])
            else:
                raise ValueError('Invalid points in quadrature rule')

        if len(self.weights) and len(self.weights) != len(self.points):
            raise ValueError('Invalid number of weights')

        # Flatten 1D rules
        if self.ndim == 1:
            self.points = [p[0] for p in self.points]

    @lazyprop
    def np_points(self):
        return np.asanyarray(self.points, dtype=np.float)

    @lazyprop
    def np_weights(self):
        return np.asanyarray(self.weights, dtype=np.float)


class BaseStoredQuadRule(BaseTabulatedQuadRule):
    @classmethod
    def _iter_rules(cls):
        for path in resource_listdir(__name__, cls.shape):
            m = re.match(r'([a-zA-Z0-9\-~+]+)-n(\d+)'
                         r'(?:-d(\d+))?(?:-([spu]+))?\.txt$', path)
            if m:
                yield (path, m.group(1), int(m.group(2)),
                       int(m.group(3) or -1), set(m.group(4) or ''))

    def __init__(self, name=None, npts=None, qdeg=None, flags=None):
        if not npts and not qdeg:
            raise ValueError('Must specify either npts or qdeg')

        best = None
        for rpath, rname, rnpts, rqdeg, rflags in self._iter_rules():
            # See if this rule fulfils the required criterion
            if ((not name or name == rname) and
                (not npts or npts == rnpts) and
                (not qdeg or qdeg <= rqdeg) and
                (not flags or set(flags) <= rflags)):
                # If so see if it is better than the current candidate
                if (not best or
                    (npts and rqdeg > best[2]) or
                    (qdeg and rnpts < best[1])):
                    best = (rpath, rnpts, rqdeg)

        # Raise if no suitable rules were found
        if not best:
            raise ValueError('No suitable quadrature rule found')

        # Load the rule
        rule = resource_string(__name__, '{}/{}'.format(self.shape, best[0]))
        super(BaseStoredQuadRule, self).__init__(rule)


def get_quadrule(eletype, rule=None, npts=None, qdeg=None, flags=None):
    ndims = dict(line=1, quad=2, tri=2, hex=3, pri=3, tet=3)

    if rule and not re.match('[a-zA-z0-9\-~+]+$', rule):
        class TabulatedQuadRule(BaseTabulatedQuadRule):
            shape = eletype
            ndim = ndims[eletype]

        r = TabulatedQuadRule(rule)

        # Validate the provided point set
        if npts and npts != len(r.points):
            raise ValueError('Invalid number of points in provided rule')

        if qdeg and not r.weights:
            raise ValueError('Provided rule has no quadrature weights')

        return r
    else:
        class StoredQuadRule(BaseStoredQuadRule):
            shape = eletype
            ndim = ndims[eletype]

        return StoredQuadRule(rule, npts, qdeg, flags)
