# -*- coding: utf-8 -*-

import itertools as it

import numpy as np
import sympy as sy
from sympy.mpmath import mp

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

    def __init__(self, *args, **kwargs):
        super(TensorProdBasis, self).__init__(*args, **kwargs)

        if self.nspts:
            # Obtain the shape point order
            nsptsord = mp.nthroot(self.nspts, self.ndims)

            if not mp.isint(nsptsord):
                raise ValueError('Invalid number of shape points for {} dims'
                                 .format(self.ndims))

            self._nsptsord = int(nsptsord)

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

    _fpairs = [(3, 1), (0, 2)]

    @lazyprop
    def fpts(self):
        # Get the 1D points
        pts1d = self._pts1d

        # Project onto the edges
        fpts = np.empty((4, len(pts1d), 2), dtype=np.object)
        fpts[0,:,0], fpts[0,:,1] = pts1d, -1
        fpts[1,:,0], fpts[1,:,1] = 1, pts1d
        fpts[2,:,0], fpts[2,:,1] = pts1d, 1
        fpts[3,:,0], fpts[3,:,1] = -1, pts1d

        return fpts.reshape(-1, 2)

    @lazyprop
    def norm_fpts(self):
        nfpts = np.empty((4, self._order + 1, 2), dtype=np.int)
        nfpts[0,:,:] = (0, -1)
        nfpts[1,:,:] = (1, 0)
        nfpts[2,:,:] = (0, 1)
        nfpts[3,:,:] = (-1, 0)

        return nfpts.reshape(-1, 2)


class HexBasis(TensorProdBasis, BaseBasis):
    name = 'hex'
    ndims = 3

    _fpairs = [(4, 2), (1, 3), (0, 5)]

    @lazyprop
    def fpts(self):
        # Flux points for a single face
        rule = self._cfg.get('solver-elements-hex', 'soln-pts')
        s, t = np.array(get_quadrule('quad', rule, self.nfpts // 6).points).T

        # Flux points
        fpts = np.empty((6, self.nfpts // 6, 3), dtype=np.object)
        fpts[0,:,0], fpts[0,:,1], fpts[0,:,2] = s, t, -1
        fpts[1,:,0], fpts[1,:,1], fpts[1,:,2] = s, -1, t
        fpts[2,:,0], fpts[2,:,1], fpts[2,:,2] = 1, s, t
        fpts[3,:,0], fpts[3,:,1], fpts[3,:,2] = s, 1, t
        fpts[4,:,0], fpts[4,:,1], fpts[4,:,2] = -1, s, t
        fpts[5,:,0], fpts[5,:,1], fpts[5,:,2] = s, t, 1

        return fpts.reshape(-1, 3)

    @lazyprop
    def norm_fpts(self):
        nfpts = np.empty((6, self.nfpts // 6, 3), dtype=np.int)
        nfpts[0,:,:] = (0, 0, -1)
        nfpts[1,:,:] = (0, -1, 0)
        nfpts[2,:,:] = (1, 0, 0)
        nfpts[3,:,:] = (0, 1, 0)
        nfpts[4,:,:] = (-1, 0, 0)
        nfpts[5,:,:] = (0, 0, 1)

        return nfpts.reshape(-1, 3)
