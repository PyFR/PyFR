# -*- coding: utf-8 -*-

from math import sqrt

import numpy as np

from pyfr.bases.base import BaseBasis
from pyfr.polys import get_polybasis
from pyfr.quadrules import get_quadrule
from pyfr.util import lazyprop


class PriBasis(BaseBasis):
    name = 'pri'
    ndims = 3

    # nspts = n^2*(n + 1)/2
    nspts_coeffs = [1, 1, 0, 0]
    nspts_cdenom = 2

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q, r)
                for r in pts1d
                for i, q in enumerate(pts1d)
                for p in pts1d[:(sptord + 1 - i)]]

    @property
    def nupts(self):
        return (self._order + 1)**2*(self._order + 2) // 2

    @lazyprop
    def fpts(self):
        n = self._order + 1

        # Tri face points
        tname = self._cfg.get('solver-interfaces-tri', 'flux-pts')
        ts, tt = get_quadrule('tri', tname, n*(n + 1) // 2).np_points.T

        # Quad face points
        qname = self._cfg.get('solver-interfaces-quad', 'flux-pts')
        qs, qt = get_quadrule('quad', qname, n**2).np_points.T

        # Project
        proj = [(ts, tt, -1), (ts, tt, 1), (qs, -1, qt), (-qs, qs, qt),
                (-1, qs, qt)]

        return np.vstack(list(np.broadcast(*p)) for p in proj)

    @property
    def facenorms(self):
        c = 1 / sqrt(2)
        return [(0, 0, -1), (0, 0, 1), (0, -1, 0), (c, c, 0), (-1, 0, 0)]

    @property
    def facefpts(self):
        n = self._order + 1

        tpts = np.arange(n*(n + 1)).reshape(2, -1)
        qpts = np.arange(3*n**2).reshape(3, -1) + n*(n + 1)

        return tpts.tolist() + qpts.tolist()

    @lazyprop
    def fbasis_coeffs(self):
        n = self._order + 1

        tfproj = lambda s, t: [(s, t, -1), (s, t, 1)]
        qfproj = lambda s, t: [(s, -1, t), (-s, s, t), (-1, s, t)]

        tS = self._fbasis_coeffs_for('tri', tfproj, [1, 1], n*(n + 1) // 2)
        qS = self._fbasis_coeffs_for('quad', qfproj, [1, sqrt(2), 1], n**2)

        return np.vstack([tS, qS])
