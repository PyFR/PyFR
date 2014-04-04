# -*- coding: utf-8 -*-

import itertools as it

import numpy as np

from pyfr.bases.base import BaseBasis
from pyfr.quadrules import get_quadrule
from pyfr.util import lazyprop


class TensorProdBasis(object):
    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)
        return list(p[::-1] for p in it.product(pts1d, repeat=cls.ndims))

    @property
    def facefpts(self):
        kn = (self._order + 1)**(self.ndims - 1)
        return [list(xrange(i*kn, (i + 1)*kn)) for i in xrange(2*self.ndims)]

    @property
    def nupts(self):
        return (self._order + 1)**self.ndims


class QuadBasis(TensorProdBasis, BaseBasis):
    name = 'quad'
    ndims = 2

    # nspts = n^2
    nspts_coeffs = [1, 0, 0]
    nspts_cdenom = 1

    @lazyprop
    def fpts(self):
        # Flux points along an edge
        rulename = self._cfg.get('solver-interfaces-line', 'flux-pts')
        pts = np.array(get_quadrule('line', rulename, self.nfpts // 4).points)

        # Project onto the edges
        proj = [(pts, -1), (1, pts), (pts, 1), (-1, pts)]

        return np.vstack(list(np.broadcast(*p)) for p in proj)

    @property
    def facenorms(self):
        return [(0, -1), (1, 0), (0, 1), (-1, 0)]

    @lazyprop
    def fbasis_coeffs(self):
        fproj = lambda pts: [(pts, -1), (1, pts), (pts, 1), (-1, pts)]

        return self._fbasis_coeffs_for('line', fproj, [1]*4, self.nfpts // 4)


class HexBasis(TensorProdBasis, BaseBasis):
    name = 'hex'
    ndims = 3

    # nspts = n^3
    nspts_coeffs = [1, 0, 0, 0]
    nspts_cdenom = 1

    @lazyprop
    def fpts(self):
        # Flux points for a single face
        rule = self._cfg.get('solver-elements-hex', 'soln-pts')
        s, t = get_quadrule('quad', rule, self.nfpts // 6).np_points.T

        # Project
        proj = [(s, t, -1), (s, -1, t), (1, s, t),
                (s, 1, t), (-1, s, t), (s, t, 1)]

        return np.vstack(list(np.broadcast(*p)) for p in proj)

    @property
    def facenorms(self):
        return [(0, 0, -1), (0, -1, 0), (1, 0, 0),
                (0, 1, 0), (-1, 0, 0), (0, 0, 1)]

    @lazyprop
    def fbasis_coeffs(self):
        fproj = lambda s, t: [(s, t, -1), (s, -1, t), (1, s, t),
                              (s, 1, t), (-1, s, t), (s, t, 1)]

        return self._fbasis_coeffs_for('quad', fproj, [1]*6, self.nfpts // 6)
