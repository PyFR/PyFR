# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
import re

import numpy as np
from sympy.mpmath import mp
from sympy.utilities.lambdify import lambdastr

from pyfr.util import lazyprop, ndrange


def lambdify_mpf(dims, exprs):
    # Perform the initial lambdification
    ls = [lambdastr(dims, ex.evalf(mp.dps)) for ex in exprs]
    csf = {}

    # Locate all numerical constants in these lambdified expressions
    for l in ls:
        for m in re.findall(r'([0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?)', l):
            if m not in csf:
                csf[m] = mp.mpf(m)

    # Sort the keys by their length to prevent erroneous substitutions
    cs = sorted(csf, key=len, reverse=True)

    # Name these constants
    csn = {s: '__c%d' % i for i, s in enumerate(cs)}
    cnf = {n: csf[s] for s, n in csn.iteritems()}

    # Substitute
    lex = []
    for l in ls:
        for s in cs:
            l = l.replace(s, csn[s])
        lex.append(eval(l, cnf))

    return lex


def lambdify_jac_mpf(dims, exprs):
    jac_exprs = [ex.diff(d) for ex in exprs for d in dims]
    return lambdify_mpf(dims, jac_exprs)


class BaseBasis(object):
    __metaclass__ = ABCMeta

    name = None
    ndims = -1

    def __init__(self, dims, nspts, cfg):
        self._dims = dims
        self._nspts = nspts
        self._cfg = cfg
        self._order = cfg.getint('solver', 'order')

        if self.ndims != len(dims):
            raise ValueError('Invalid dimension symbols')

    @property
    def dims(self):
        return self._dims

    @abstractmethod
    def std_ele(sptord):
        pass

    @lazyprop
    def m0(self):
        """Discontinuous soln at upts to discontinuous soln at fpts"""
        return self.ubasis_at(self.fpts)

    @lazyprop
    def m1(self):
        """Trans discontinuous flux at upts to trans divergence of
        trans discontinuous flux at upts
        """
        return self.jac_ubasis_at(self.upts)

    @lazyprop
    def m2(self):
        """Trans discontinuous flux at upts to trans normal
        discontinuous flux at fpts
        """
        return self.norm_fpts[:,None,:]*self.m0[...,None]

    @lazyprop
    def m3(self):
        """Trans normal correction flux at upts to trans divergence of
        trans correction flux at upts
        """
        return self.fbasis_at(self.upts)

    @property
    def m132(self):
        m1, m2, m3 = self.m1, self.m2, self.m3
        return m1 - np.dot(m3, m2.reshape(self.nfpts, -1)).reshape(m1.shape)

    @lazyprop
    def m4(self):
        """Discontinuous soln at upts to trans gradient of discontinuous
        solution at upts
        """
        return self.m1.swapaxes(2, 1)[...,None]

    @lazyprop
    def m5(self):
        """Trans grad discontinuous soln at upts to trans gradient of
        discontinuous solution at fpts
        """
        nfpts, ndims, nupts = self.nfpts, self.ndims, self.nupts
        m = np.zeros((nfpts, ndims, nupts, ndims), dtype=self.m0.dtype)

        for i in xrange(ndims):
            m[:,i,:,i] = self.m0

        return m

    @lazyprop
    def m6(self):
        """Correction soln at fpts to trans gradient of correction
        solution at upts
        """
        m = self.norm_fpts.T[:,None,:]*self.m3
        return m.swapaxes(0, 1)[...,None]

    @property
    def m460(self):
        m4, m6, m0 = self.m4, self.m6, self.m0
        return m4 - np.dot(m6.reshape(-1, self.nfpts), m0).reshape(m4.shape)

    @property
    def nspts(self):
        return self._nspts

    def _eval_lbasis_at(self, lbasis, pts):
        m = np.empty((len(pts), len(lbasis)), dtype=np.object)

        for i, j in ndrange(*m.shape):
            m[i,j] = lbasis[j](*pts[i])

        m[abs(m) < 1e-14] = 0
        return m

    def _eval_jac_lbasis_at(self, jlbasis, pts):
        m = self._eval_lbasis_at(jlbasis, pts)
        return m.reshape(len(pts), -1, self.ndims)

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
    def facefpts(self):
        pass

    @lazyprop
    def nfacefpts(self):
        return [len(f) for f in self.facefpts]

    @property
    def nfpts(self):
        return sum(self.nfacefpts)
