# -*- coding: utf-8 -*-

from pkg_resources import resource_listdir, resource_string
import re

import numpy as np


class BaseTabulatedQuadRule(object):
    def __init__(self, rule):
        pts = []
        wts = []

        rule = re.sub(r'(?<=\))\s*,?\s*(?!$)', r'\n', rule)
        rule = re.sub(r'\(|\)|,', '', rule).strip()
        rule = rule[1:-1] if rule.startswith('[') else rule

        for l in rule.splitlines():
            if not l:
                continue

            # Parse the line
            args = [float(f) for f in l.split()]

            if len(args) == self.ndim:
                pts.append(args)
            elif len(args) == self.ndim + 1:
                pts.append(args[:-1])
                wts.append(args[-1])
            else:
                raise ValueError('Invalid points in quadrature rule')

        if len(wts) and len(wts) != len(pts):
            raise ValueError('Invalid number of weights')

        # Flatten 1D rules
        if self.ndim == 1:
            pts = [p[0] for p in pts]

        # Cast
        self.pts = np.array(pts, dtype=np.float)
        self.wts = np.array(wts, dtype=np.float)


class BaseStoredQuadRule(BaseTabulatedQuadRule):
    @classmethod
    def _iter_rules(cls):
        rpaths = getattr(cls, '_rpaths', None)
        if rpaths is None:
            cls._rpaths = rpaths = resource_listdir(__name__, cls.shape)

        for path in rpaths:
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
        super().__init__(rule.decode('utf-8'))


def get_quadrule(eletype, rule=None, npts=None, qdeg=None, flags=None):
    ndims = dict(line=1, quad=2, tri=2, hex=3, pri=3, pyr=3, tet=3)

    if rule and not re.match(r'[a-zA-z0-9\-~+]+$', rule):
        class TabulatedQuadRule(BaseTabulatedQuadRule):
            shape = eletype
            ndim = ndims[eletype]

        r = TabulatedQuadRule(rule)

        # Validate the provided point set
        if npts and npts != len(r.pts):
            raise ValueError('Invalid number of points in provided rule')

        if qdeg and not len(r.wts):
            raise ValueError('Provided rule has no quadrature weights')

        return r
    else:
        class StoredQuadRule(BaseStoredQuadRule):
            shape = eletype
            ndim = ndims[eletype]

        return StoredQuadRule(rule, npts, qdeg, flags)
