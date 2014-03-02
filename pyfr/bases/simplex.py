# -*- coding: utf-8 -*-

from math import sqrt

import numpy as np

from pyfr.bases.base import BaseBasis
from pyfr.polys import get_polybasis
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
        pts1d = get_quadrule('line', 'equi-spaced', sptord + 1).points
        sele = [(p, q)
                for i, q in enumerate(pts1d)
                for p in pts1d[:(sptord + 1 - i)]]

        return np.array(sele, dtype=np.object)

    @property
    def nupts(self):
        return (self._order + 1)*(self._order + 2) // 2

    @lazyprop
    def fpts(self):
        # 1D points
        qrule = self._cfg.get('solver-interfaces-line', 'flux-pts')
        pts1d = get_quadrule('line', qrule, self._order + 1).np_points

        # Project
        proj = [(pts1d, -1), (-pts1d, pts1d), (-1, pts1d)]

        return np.vstack(list(np.broadcast(*p)) for p in proj)

    @property
    def facenorms(self):
        return [(0, -1), (1 / sqrt(2), 1 / sqrt(2)), (-1, 0)]

    @property
    def facefpts(self):
        k = self._order + 1
        return [list(xrange(i*k, (i + 1)*k)) for i in xrange(3)]

    @lazyprop
    def fbasis_coeffs(self):
        qrule = get_quadrule('line', 'gauss-legendre', self._order + 1)
        qpts = qrule.np_points

        proj = [(qpts, -1), (-qpts, qpts), (-1, qpts)]
        qedgepts = np.vstack(list(np.broadcast(*p)) for p in proj)

        rulename = self._cfg.get('solver-interfaces-line', 'flux-pts')
        pts = get_quadrule('line', rulename, self.nfpts // 3).np_points
        nbedge = get_polybasis('line', self._order + 1, pts)

        L = nbedge.nodal_basis_at(qpts)
        M = self.ubasis.ortho_basis_at(qedgepts).reshape(-1, 3, len(qpts))

        # Do the quadrature
        S = np.einsum('i...,ik,jli->lkj', qrule.np_weights, L, M)

        # Account for the longer length of the hypotenuse
        S[1] *= sqrt(2)

        return S.reshape(-1, self.nupts)


class TetBasis(BaseBasis):
    name = 'tet'
    ndims = 3

    # nspts = n*(n + 1)*(n + 2)/6
    nspts_coeffs = [1, 3, 2, 0]
    nspts_cdenom = 6

    @classmethod
    def std_ele(cls, sptord):
        pts1d = get_quadrule('line', 'equi-spaced', sptord + 1).points
        sele = [(p, q, r)
                for i, r in enumerate(pts1d)
                for j, q in enumerate(pts1d[:(sptord + 1 - i)])
                for p in pts1d[:(sptord + 1 - i - j)]]

        return np.array(sele, dtype=np.object)

    @property
    def nupts(self):
        return (self._order + 1)*(self._order + 2)*(self._order + 3) // 6

    @lazyprop
    def fpts(self):
        # 2D points on a triangle
        qrule = self._cfg.get('solver-interfaces-tri', 'flux-pts')
        npts2d = (self._order + 1)*(self._order + 2) // 2

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
        n = (self._order + 1)*(self._order + 2) // 2
        return [list(xrange(i*n, (i + 1)*n)) for i in xrange(4)]

    @lazyprop
    def fbasis_coeffs(self):
        # Obtain a high-power quadrature rule on the reference
        # triangle; this is sufficient up to order 6
        qrule = get_quadrule('tri', 'williams-shunn', 36)
        s, t = qrule.np_points.T

        proj = [(s, t, -1), (s, -1, t), (-1, t, s), (s, t, -s -t -1)]
        qedgepts = np.vstack(list(np.broadcast(*p)) for p in proj)

        rulename = self._cfg.get('solver-interfaces-tri', 'flux-pts')
        pts = get_quadrule('tri', rulename, self.nfpts // 4).np_points
        nbedge = get_polybasis('tri', self._order + 1, pts)

        L = nbedge.nodal_basis_at(qrule.np_points)
        M = self.ubasis.ortho_basis_at(qedgepts).reshape(-1, 4, len(s))

        # Do the quadrature
        S = np.einsum('i...,ik,jli->lkj', qrule.np_weights, L, M)

        # Account for the greater area of the centre face
        S[3] *= sqrt(3)

        return S.reshape(-1, self.nupts)
