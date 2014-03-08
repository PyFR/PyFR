# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty

from mpmath import mp
import numpy as np

from pyfr.nputil import chop
from pyfr.polys import get_polybasis
from pyfr.quadrules import get_quadrule
from pyfr.util import lazyprop


class BaseBasis(object):
    __metaclass__ = ABCMeta

    name = None
    ndims = -1

    nspts_coeffs = None
    nspts_cdenom = None

    def __init__(self, nspts, cfg):
        self._nspts = nspts
        self._cfg = cfg
        self._order = cfg.getint('solver', 'order')

        self.ubasis = get_polybasis(self.name, self._order + 1, self.upts)

        if nspts:
            self._nsptsord = nsptord = self.order_from_nspts(nspts)
            self.sbasis = get_polybasis(self.name, nsptord, self.spts)

    @abstractmethod
    def std_ele(sptord):
        pass

    @classmethod
    def nspts_from_order(cls, sptord):
        return int(mp.polyval(cls.nspts_coeffs, sptord)) / cls.nspts_cdenom

    @classmethod
    def order_from_nspts(cls, nspts):
        # Obtain the coefficients for the poly: P(n) - nspts = 0
        coeffs = list(cls.nspts_coeffs)
        coeffs[-1] -= cls.nspts_cdenom*nspts

        # Solve to obtain the order (a positive integer)
        roots = mp.polyroots(coeffs)
        roots = [int(x) for x in roots if mp.isint(x) and x > 0]

        if roots:
            return roots[0]
        else:
            raise ValueError('Invalid number of shape points')

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

    @abstractproperty
    def nupts(self):
        pass

    @lazyprop
    def upts(self):
        rname = self._cfg.get('solver-elements-' + self.name, 'soln-pts')
        return get_quadrule(self.name, rname, self.nupts).points

    def ubasis_at(self, pts):
        return self.ubasis.nodal_basis_at(pts)

    def jac_ubasis_at(self, pts):
        return np.rollaxis(self.ubasis.jac_nodal_basis_at(pts), 2)

    @abstractproperty
    def fpts(self):
        pass

    @abstractproperty
    def fbasis_coeffs(self):
        pass

    def _fbasis_coeffs_for(self, ftype, fproj, fdjacs, nffpts):
        # Suitable quadrature rules for various face types
        qrule_map = {
            'line': ('gauss-legendre', self._order + 1),
            'quad': ('gauss-legendre', (self._order + 1)**2),
            'tri': ('williams-shunn', 36)
        }

        # Obtain a quadrature rule for integrating on the face
        qrule = get_quadrule(ftype, *qrule_map[ftype])

        # Project the rule points onto the various faces
        proj = fproj(*np.atleast_2d(qrule.np_points.T))
        qfacepts = np.vstack(list(np.broadcast(*p)) for p in proj)

        # Obtain a nodal basis on the reference face
        fname = self._cfg.get('solver-interfaces-' + ftype, 'flux-pts')
        ffpts = get_quadrule(ftype, fname, nffpts)
        nodeb = get_polybasis(ftype, self._order + 1, ffpts.np_points)

        L = nodeb.nodal_basis_at(qrule.np_points)

        M = self.ubasis.ortho_basis_at(qfacepts)
        M = M.reshape(-1, len(proj), len(qrule.np_points))

        # Do the quadrature
        S = np.einsum('i...,ik,jli->lkj', qrule.np_weights, L, M)

        # Account for differing face areas
        S *= np.asanyarray(fdjacs)[:,None,None]

        return S.reshape(-1, self.nupts)

    @chop
    def fbasis_at(self, pts):
        return np.dot(self.fbasis_coeffs, self.ubasis.ortho_basis_at(pts)).T

    @abstractproperty
    def facenorms(self):
        pass

    @lazyprop
    def norm_fpts(self):
        fnorms = self.facenorms
        return np.vstack([fn]*n for fn, n in zip(fnorms, self.nfacefpts))

    @lazyprop
    def spts(self):
        return self.std_ele(self._nsptsord - 1)

    def sbasis_at(self, pts):
        return self.sbasis.nodal_basis_at(pts)

    def jac_sbasis_at(self, pts):
        return np.rollaxis(self.sbasis.jac_nodal_basis_at(pts), 2)

    @abstractproperty
    def facefpts(self):
        pass

    @lazyprop
    def nfacefpts(self):
        return [len(f) for f in self.facefpts]

    @property
    def nfpts(self):
        return sum(self.nfacefpts)
