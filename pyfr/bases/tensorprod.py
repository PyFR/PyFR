# -*- coding: utf-8 -*-

import numpy as np

from pyfr.bases.base import BaseBasis
from pyfr.polys import get_polybasis
from pyfr.quadrules import get_quadrule
from pyfr.util import lazyprop


class TensorProdBasis(object):
    @classmethod
    def std_ele(cls, sptord):
        n = (sptord + 1)**cls.ndims
        return get_quadrule(cls.name, 'equi-spaced', n).points

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
        qrule = get_quadrule('line', 'gauss-legendre', self._order + 1)
        qpts = qrule.np_points

        proj = [(qpts, -1), (1, qpts), (qpts, 1), (-1, qpts)]
        qedgepts = np.vstack(list(np.broadcast(*p)) for p in proj)

        rulename = self._cfg.get('solver-interfaces-line', 'flux-pts')
        pts = get_quadrule('line', rulename, self.nfpts // 4).np_points
        nbedge = get_polybasis('line', self._order + 1, pts)

        L = nbedge.nodal_basis_at(qpts)
        M = self.ubasis.ortho_basis_at(qedgepts).reshape(-1, 4, len(qpts))

        # Do the quadrature
        S = np.einsum('i...,ik,jli->lkj', qrule.np_weights, L, M)

        return S.reshape(-1, self.nupts)


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
        qrule = get_quadrule('quad', 'gauss-legendre', (self._order + 1)**2)
        s, t = qrule.np_points.T

        proj = [(s, t, -1), (s, -1, t), (1, s, t),
                (s, 1, t), (-1, s, t), (s, t, 1)]
        qfacepts = np.vstack(list(np.broadcast(*p)) for p in proj)

        rulename = self._cfg.get('solver-interfaces-quad', 'flux-pts')
        fptsrule = get_quadrule('quad', rulename, self.nfpts // 6)
        nbface = get_polybasis('quad', self._order + 1, fptsrule.points)

        L = nbface.nodal_basis_at(qrule.np_points)
        M = self.ubasis.ortho_basis_at(qfacepts).reshape(-1, 6, len(s))

        # Do the quadrature
        S = np.einsum('i...,ik,jli->lkj', qrule.np_weights, L, M)

        return S.reshape(-1, self.nupts)
