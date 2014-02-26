# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from pyfr.quadrules import get_quadrule
from pyfr.syutil import lambdify_jac_mpf, lambdify_mpf
from pyfr.util import lazyprop, ndrange


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
        return self.jac_ubasis_at(self.upts).reshape(self.nupts, -1)

    @lazyprop
    def m2(self):
        """Trans discontinuous flux at upts to trans normal
        discontinuous flux at fpts
        """
        m = self.norm_fpts[...,None]*self.m0[:,None,:]
        return m.reshape(self.nfpts, -1)

    @lazyprop
    def m3(self):
        """Trans normal correction flux at upts to trans divergence of
        trans correction flux at upts
        """
        return self.fbasis_at(self.upts)

    @property
    def m132(self):
        return self.m1 - np.dot(self.m3, self.m2)

    @lazyprop
    def m4(self):
        """Discontinuous soln at upts to trans gradient of discontinuous
        solution at upts
        """
        m = self.m1.reshape(self.nupts, -1, self.nupts).swapaxes(0, 1)
        return m.reshape(-1, self.nupts)

    @lazyprop
    def m6(self):
        """Correction soln at fpts to trans gradient of correction
        solution at upts
        """
        m = self.norm_fpts.T[:,None,:]*self.m3
        return m.reshape(-1, self.nfpts)

    @property
    def m460(self):
        return self.m4 - np.dot(self.m6, self.m0)

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
        return m.reshape(len(pts), -1, self.ndims).swapaxes(1, 2)

    @abstractproperty
    def nupts(self):
        pass

    @lazyprop
    def upts(self):
        rname = self._cfg.get('solver-elements-' + self.name, 'soln-pts')
        return get_quadrule(self.name, rname, self.nupts).points

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

    @lazyprop
    def spts(self):
        return self.std_ele(self._nsptsord - 1)

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
