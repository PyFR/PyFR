# -*- coding: utf-8 -*-

import numpy as np
import sympy as sy
from sympy.mpmath import mp

from pyfr.bases.base import BaseBasis
from pyfr.quadrules import BaseLineQuadRule, BaseTriQuadRule, get_quadrule
from pyfr.syutil import lagrange_basis, nodal_basis
from pyfr.util import lazyprop, memoize


def _bary_to_cart(b, tverts):
    return [sum(bj*ej for bj, ej in zip(b, t)) for t in zip(*tverts)]


class TriBasis(BaseBasis):
    name = 'tri'
    ndims = 2

    def __init__(self, *args, **kwargs):
        super(TriBasis, self).__init__(*args, **kwargs)

        if self.nspts:
            # Solve nspts = n*(n+1)/2 for n to get the shape point order
            self._nsptsord = (sy.sqrt(8*self.nspts + 1) - 1) / 2

            if not self._nsptsord.is_Number:
                raise ValueError('Invalid number of shape points')

    @classmethod
    def std_ele(cls, sptord):
        esqr = get_quadrule(BaseLineQuadRule, 'equi-spaced', sptord + 1)
        sele = [(p, q)
                for i, q in enumerate(esqr.points)
                for p in esqr.points[:(sptord + 1 - i)]]

        return np.array(sele, dtype=np.object)

    @memoize
    def _orthonormal_basis(self, ptsord):
        p, q = self._dims
        a, b = 2*(1 + p)/(1 - q) - 1, q

        # Construct an orthonormal basis within a standard triangle
        db = []
        for i in xrange(ptsord):
            tmp = sy.sqrt(2)*sy.jacobi_normalized(i, 0, 0, a)*(1 - b)**i
            tmp = tmp.ratsimp()

            for j in xrange(ptsord - i):
                poly = sy.expand(tmp*sy.jacobi_normalized(j, 2*i + 1, 0, b))
                db.append(poly.evalf(mp.dps))

        return db

    @lazyprop
    def upts(self):
        qrule = self._cfg.get('solver-elements-tri', 'soln-pts')
        bupts = get_quadrule(BaseTriQuadRule, qrule, self.nupts).points

        # Convert to Cartesian
        stdtri = self.std_ele(1)
        return np.array([_bary_to_cart(b, stdtri) for b in bupts],
                         dtype=np.object)

    @property
    def nupts(self):
        return (self._order + 1)*(self._order + 2) // 2

    @lazyprop
    def ubasis(self):
        return nodal_basis(self.upts,
                           self._orthonormal_basis(self._order + 1),
                           dims=self._dims)

    @lazyprop
    def spts(self):
        return self.std_ele(self._nsptsord - 1)

    @lazyprop
    def sbasis(self):
        return nodal_basis(self.spts,
                           self._orthonormal_basis(self._nsptsord),
                           dims=self._dims)

    @lazyprop
    def fpts(self):
        # 1D points
        qrule = self._cfg.get('solver-interfaces-line', 'flux-pts')
        pts1d = get_quadrule(BaseLineQuadRule, qrule, self._order + 1).points

        # Flux points
        fpts = np.empty((3, self._order + 1, 2), dtype=np.object)
        fpts[0,:,0], fpts[0,:,1] = pts1d, -1
        fpts[1,:,0], fpts[1,:,1] = pts1d[::-1], pts1d
        fpts[2,:,0], fpts[2,:,1] = -1, pts1d[::-1]

        return fpts.reshape(-1, 2)

    @lazyprop
    def norm_fpts(self):
        nfpts = np.empty((3, self._order + 1, 2), dtype=np.object)
        nfpts[0,:,:] = (0, -1)
        nfpts[1,:,:] = (1/mp.sqrt(2), 1/mp.sqrt(2))
        nfpts[2,:,:] = (-1, 0)

        return nfpts.reshape(-1, 2)

    @property
    def facefpts(self):
        k = self._order + 1
        return [list(xrange(i*k, (i + 1)*k)) for i in xrange(3)]

    @lazyprop
    def fbasis(self):
        # Dummy parametric symbol
        t = sy.Symbol('t')

        # Dimension variables
        p, q = self._dims

        # Orthonormal basis
        obasis = self._orthonormal_basis(self._order + 1)

        # Nodal basis along an edge
        qrule = self._cfg.get('solver-interfaces-line', 'flux-pts')
        pts1d = get_quadrule(BaseLineQuadRule, qrule, self._order + 1).points
        nb1d = lagrange_basis(pts1d, t)

        # Allocate space for the flux point basis
        fbasis = np.empty((3, len(nb1d)), dtype=np.object)

        # Parametric mappings (p,q) -> t for the three edges
        # (bottom, hypot, left)
        substs = [{p: t, q: -1}, {p: -t, q: t}, {p: -1, q: -t}]

        for i, esub in enumerate(substs):
            for j, lj in enumerate(nb1d):
                fb = sum(sy.integrate(lj*ob.subs(esub), (t, -1, 1))*ob
                         for ob in obasis)
                fbasis[i,j] = fb

        # Account for the longer length of the hypotenuse
        fbasis[1,:] *= mp.sqrt(2)

        return fbasis.ravel()
