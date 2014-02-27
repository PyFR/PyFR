# -*- coding: utf-8 -*-

import itertools as it

import numpy as np
import sympy as sy

from pyfr.bases.base import BaseBasis
from pyfr.quadrules import get_quadrule
from pyfr.syutil import lagrange_basis
from pyfr.util import lazyprop


def nodal_basis(points, dims, compact=True):
    p = list(points)

    # Evaluate the basis function in terms of each dimension
    basis = [lagrange_basis(p, d) for d in reversed(dims)]

    # Take the cartesian product of these and multiply the resulting tuples
    cpbasis = np.array([np.prod(b) for b in it.product(*basis)])

    return cpbasis if compact else cpbasis.reshape((len(p),)*len(dims))


class TensorProdBasis(object):
    # List of face numbers paired according to their normal dimension
    # e.g, [(a, b), ...] where a, b are the faces whose normal points
    # in -p and p, respectively
    _fpairs = None

    @classmethod
    def std_ele(cls, sptord):
        n = (sptord + 1)**cls.ndims
        return get_quadrule(cls.name, 'equi-spaced', n).points

    @lazyprop
    def _pts1d(self):
        rule = self._cfg.get('solver-elements-' + self.name, 'soln-pts')
        return get_quadrule('line', rule, self._order + 1).points

    def _vcjh_fn(self, sym):
        k = self._order
        eta = self._cfg.get('solver-elements-' + self.name, 'vcjh-eta')

        # Expand shorthand forms of eta for common schemes
        etacommon = dict(dg='0', sd='k/(k+1)', hu='(k+1)/k')
        eta_k = sy.S(etacommon.get(eta, eta), locals=dict(k=k))

        lkm1, lk, lkp1 = [sy.legendre_poly(m, sym) for m in [k - 1, k, k + 1]]
        return sy.S(1)/2 * (lk + (eta_k*lkm1 + lkp1)/(1 + eta_k))

    @lazyprop
    def ubasis(self):
        return nodal_basis(self._pts1d, self._dims)

    @lazyprop
    def fbasis(self):
        # Get the 1D points
        pts1d = self._pts1d

        # Dummy symbol
        _x = sy.Symbol('_x')

        # Get the derivative of the 1D correction function
        diffg = self._vcjh_fn(_x).diff()

        # Allocate space for the flux points basis
        fbasis = np.empty([2*self.ndims] + [len(pts1d)]*(self.ndims - 1),
                          dtype=np.object)

        # Pair up opposite faces with their associated (normal) dimension
        for (fl, fr), sym in zip(self._fpairs, self._dims):
            nbdims = [d for d in self._dims if d is not sym]
            fbasis[(fl, fr),...] = nodal_basis(pts1d, nbdims, compact=False)

            fbasis[fl,...] *= diffg.subs(_x, -sym)
            fbasis[fr,...] *= diffg.subs(_x, sym)

        return fbasis.ravel()

    @property
    def facefpts(self):
        kn = (self._order + 1)**(self.ndims - 1)
        return [list(xrange(i*kn, (i + 1)*kn)) for i in xrange(2*self.ndims)]

    @lazyprop
    def sbasis(self):
        pts1d = get_quadrule('line', 'equi-spaced', self._nsptsord).points
        return nodal_basis(pts1d, self._dims)

    @property
    def nupts(self):
        return (self._order + 1)**self.ndims


class QuadBasis(TensorProdBasis, BaseBasis):
    name = 'quad'
    ndims = 2

    # nspts = n^2
    nspts_coeffs = [1, 0, 0]
    nspts_cdenom = 1

    _fpairs = [(3, 1), (0, 2)]

    @lazyprop
    def fpts(self):
        # Get the 1D points
        pts1d = self._pts1d

        # Project onto the edges
        proj = [(pts1d, -1), (1, pts1d), (pts1d, 1), (-1, pts1d)]

        return np.vstack([list(np.broadcast(*p)) for p in proj])

    @property
    def facenorms(self):
        return [(0, -1), (1, 0), (0, 1), (-1, 0)]


class HexBasis(TensorProdBasis, BaseBasis):
    name = 'hex'
    ndims = 3

    # nspts = n^3
    nspts_coeffs = [1, 0, 0, 0]
    nspts_cdenom = 1

    _fpairs = [(4, 2), (1, 3), (0, 5)]

    @lazyprop
    def fpts(self):
        # Flux points for a single face
        rule = self._cfg.get('solver-elements-hex', 'soln-pts')
        s, t = np.array(get_quadrule('quad', rule, self.nfpts // 6).points).T

        # Project
        proj = [(s, t, -1), (s, -1, t), (1, s, t),
                (s, 1, t), (-1, s, t), (s, t, 1)]

        return np.vstack([list(np.broadcast(*p)) for p in proj])

    @property
    def facenorms(self):
        return [(0, 0, -1), (0, -1, 0), (1, 0, 0),
                (0, 1, 0), (-1, 0, 0), (0, 0, 1)]
