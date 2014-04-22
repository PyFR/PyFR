# -*- coding: utf-8 -*-

from math import sqrt

import numpy as np

from pyfr.bases.base import BaseBasis
from pyfr.quadrules import get_quadrule
from pyfr.util import lazyprop


class TriBasis(BaseBasis):
    name = 'tri'
    ndims = 2

    # nspts = n*(n + 1)/2
    nspts_coeffs = [1, 1, 0]
    nspts_cdenom = 2

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q)
                for i, q in enumerate(pts1d)
                for p in pts1d[:(sptord + 1 - i)]]

    @property
    def nupts(self):
        return (self.order + 1)*(self.order + 2) // 2

    @lazyprop
    def fpts(self):
        # 1D points
        qrule = self.cfg.get('solver-interfaces-line', 'flux-pts')
        pts1d = get_quadrule('line', qrule, self.order + 1).np_points

        # Project
        proj = [(pts1d, -1), (-pts1d, pts1d), (-1, pts1d)]

        return np.vstack(list(np.broadcast(*p)) for p in proj)

    @property
    def facenorms(self):
        return [(0, -1), (1 / sqrt(2), 1 / sqrt(2)), (-1, 0)]

    @property
    def facefpts(self):
        k = self.order + 1
        return [list(xrange(i*k, (i + 1)*k)) for i in xrange(3)]

    @lazyprop
    def fbasis_coeffs(self):
        fproj = lambda pts: [(pts, -1), (-pts, pts), (-1, pts)]

        return self._fbasis_coeffs_for('line', fproj, [1, sqrt(2), 1],
                                       self.nfpts // 3)


class TetBasis(BaseBasis):
    name = 'tet'
    ndims = 3

    # nspts = n*(n + 1)*(n + 2)/6
    nspts_coeffs = [1, 3, 2, 0]
    nspts_cdenom = 6

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q, r)
                for i, r in enumerate(pts1d)
                for j, q in enumerate(pts1d[:(sptord + 1 - i)])
                for p in pts1d[:(sptord + 1 - i - j)]]

    @property
    def nupts(self):
        return (self.order + 1)*(self.order + 2)*(self.order + 3) // 6

    @lazyprop
    def fpts(self):
        # 2D points on a triangle
        qrule = self.cfg.get('solver-interfaces-tri', 'flux-pts')
        npts2d = (self.order + 1)*(self.order + 2) // 2

        s, t = get_quadrule('tri', qrule, npts2d).np_points.T

        # Project
        proj = [(s, t, -1), (s, -1, t), (-1, t, s), (s, t, -s -t -1)]

        return np.vstack(list(np.broadcast(*p)) for p in proj)

    @property
    def facenorms(self):
        c = 1 / sqrt(3)
        return [(0, 0, -1), (0, -1, 0), (-1, 0, 0), (c, c, c)]

    @property
    def facefpts(self):
        n = (self.order + 1)*(self.order + 2) // 2
        return [list(xrange(i*n, (i + 1)*n)) for i in xrange(4)]

    @lazyprop
    def fbasis_coeffs(self):
        fproj = lambda s, t: [(s, t, -1), (s, -1, t), (-1, t, s),
                              (s, t, -s -t -1)]

        return self._fbasis_coeffs_for('tri', fproj, [1, 1, 1, sqrt(3)],
                                       self.nfpts // 4)
