# -*- coding: utf-8 -*-

import re

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from sympy.utilities.lambdify import lambdastr

try:
    from mpmath import mp
except ImportError:
    from sympy.mpmath import mp

from pyfr.util import lazyprop, ndrange

def lambdify_mpf(dims, exprs):
    # Perform the initial lambdification
    ls = [lambdastr(dims, ex.evalf(mp.dps)) for ex in exprs]
    csf = {}

    # Locate all numerical constants in these lambdified expressions
    for l in ls:
        for m in re.findall('([0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?)', l):
            if m not in csf:
                csf[m] = mp.mpf(m)

    # Name these constants
    csn = {s: '__c%d' % i for i,s in enumerate(csf.iterkeys())}
    cnf = {n: csf[s] for s,n in csn.iteritems()}

    # Substitute
    lex = []
    for l in ls:
        for s,n in csn.iteritems():
            l = l.replace(s, n)
        lex.append(eval(l, cnf))

    return lex

def lambdify_jac_mpf(dims, exprs):
    jac_exprs = [ex.diff(d) for ex in exprs for d in dims]
    return lambdify_mpf(dims, jac_exprs)


class BasisBase(object):
    __metaclass__ = ABCMeta

    name = None
    ndims = -1

    def __init__(self, dims, nspts, cfg):
        self._dims = dims
        self._nspts = nspts
        self._cfg = cfg
        self._order = cfg.getint('mesh-elements', 'order')

        if self.ndims != len(dims):
            raise ValueError('Invalid dimension symbols')

    @property
    def dims(self):
        return self._dims

    @property
    def nspts(self):
        return self._nspts

    def _eval_lbasis_at(self, lbasis, pts):
        m = np.empty((len(pts), len(lbasis)), dtype=np.object)

        for i,j in ndrange(*m.shape):
            m[i,j] = lbasis[j](*pts[i])

        m[abs(m) < 1e-14] = 0
        return m

    def _eval_jac_lbasis_at(self, jlbasis, pts):
        npts, nbasis = len(pts), len(jlbasis)
        m = np.empty((npts, nbasis), dtype=np.object)

        for i,j in ndrange(*m.shape):
            m[i,j] = jlbasis[j](*pts[i])

        m[abs(m) < 1e-14] = 0
        return m.reshape(npts, -1, self.ndims)

    @abstractproperty
    def nupts(self):
        pass

    @abstractproperty
    def upts(self):
        pass

    @abstractproperty
    def ubasis(self):
        pass

    @lazyprop
    def _ubasis_lamb(self):
        return lambdify_mpf(self._dims, self.ubasis)

    @lazyprop
    def _jac_ubasis_lamb(self):
        return lambdify_jac_mpf(self._dims, self.ubasis)

    def ubasis_at(self, pts):
        return self._eval_lbasis_at(self._ubasis_lamb, pts)

    def jac_ubasis_at(self, pts):
        return self._eval_jac_lbasis_at(self._jac_ubasis_lamb, pts)

    @abstractproperty
    def fpts(self):
        pass

    @abstractproperty
    def fbasis(self):
        pass

    @lazyprop
    def _fbasis_lamb(self):
        return lambdify_mpf(self._dims, self.fbasis)

    def fbasis_at(self, pts):
        return self._eval_lbasis_at(self._fbasis_lamb, pts)

    @abstractproperty
    def norm_fpts(self):
        pass

    @abstractproperty
    def spts(self):
        pass

    @abstractproperty
    def sbasis(self):
        pass

    @lazyprop
    def _sbasis_lamb(self):
        return lambdify_mpf(self._dims, self.sbasis)

    @lazyprop
    def _jac_sbasis_lamb(self):
        return lambdify_jac_mpf(self._dims, self.sbasis)

    def sbasis_at(self, pts):
        return self._eval_lbasis_at(self._sbasis_lamb, pts)

    def jac_sbasis_at(self, pts):
        return self._eval_jac_lbasis_at(self._jac_sbasis_lamb, pts)

    @abstractproperty
    def nfpts(self):
        pass

    @abstractmethod
    def fpts_idx_for_face(self, face, rtag):
        pass
