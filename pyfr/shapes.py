# -*- coding: utf-8 -*-

import itertools as it
from math import sqrt
import re

from mpmath import mp
import numpy as np

from pyfr.nputil import chop
from pyfr.polys import get_polybasis
from pyfr.quadrules import get_quadrule
from pyfr.nputil import block_diag
from pyfr.util import lazyprop


def _proj_rule_pts(projector, qrule):
    pts = np.atleast_2d(qrule.np_points.T)
    return np.vstack(np.broadcast_arrays(*projector(*pts))).T


class BaseShape(object):
    name = None
    ndims = -1

    nspts_coeffs = None
    nspts_cdenom = None

    npts_for_face = {
        'line': lambda order: order + 1,
        'tri': lambda order: (order + 1)*(order + 2) // 2,
        'quad': lambda order: (order + 1)**2
    }

    def __init__(self, nspts, cfg):
        self.nspts = nspts
        self.cfg = cfg
        self.order = cfg.getint('solver', 'order')

        self.antialias = cfg.get('solver', 'anti-alias', 'none')
        self.antialias = set(s.strip() for s in self.antialias.split(','))
        self.antialias.discard('none')
        if self.antialias - {'flux', 'div-flux'}:
            raise ValueError('Invalid anti-alias options')

        self.ubasis = get_polybasis(self.name, self.order + 1, self.upts)

        if nspts:
            self.nsptsord = nsptord = self.order_from_nspts(nspts)
            self.sbasis = get_polybasis(self.name, nsptord, self.spts)

    @classmethod
    def nspts_from_order(cls, sptord):
        return int(mp.polyval(cls.nspts_coeffs, sptord)) // cls.nspts_cdenom

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

    @chop
    def opmat(self, expr):
        if not re.match(r'[M0-9\-+*() ]+$', expr):
            raise ValueError('Invalid operator matrix expression')

        mats = {m: np.asmatrix(getattr(self, m.lower()))
                for m in re.findall(r'M\d+', expr)}

        return np.asarray(eval(expr, {'__builtins__': None}, mats))

    @lazyprop
    def m0(self):
        return self.ubasis.nodal_basis_at(self.fpts)

    @lazyprop
    def m1(self):
        m = np.rollaxis(self.ubasis.jac_nodal_basis_at(self.upts), 2)
        return m.reshape(self.nupts, -1)

    @lazyprop
    def m2(self):
        m = self.norm_fpts[...,None]*self.m0[:,None,:]
        return m.reshape(self.nfpts, -1)

    @lazyprop
    def m3(self):
        return self.fbasis_at(self.upts)

    @lazyprop
    def m4(self):
        m = self.m1.reshape(self.nupts, -1, self.nupts).swapaxes(0, 1)
        return m.reshape(-1, self.nupts)

    @lazyprop
    def m6(self):
        m = self.norm_fpts.T[:,None,:]*self.m3
        return m.reshape(-1, self.nfpts)

    @lazyprop
    def m7(self):
        return self.ubasis.nodal_basis_at(self.qpts)

    @lazyprop
    def m8(self):
        return np.vstack([self.m0, self.m7])

    @lazyprop
    @chop
    def m9(self):
        ub = self.ubasis
        return np.dot(ub.vdm.T, self.qwts*ub.ortho_basis_at(self.qpts))

    @property
    def m10(self):
        return block_diag([self.m9]*self.ndims)

    @lazyprop
    def upts(self):
        rname = self.cfg.get('solver-elements-' + self.name, 'soln-pts')
        return get_quadrule(self.name, rname, self.nupts).points

    @lazyprop
    def _qrule(self):
        sect = 'solver-elements-' + self.name
        kwargs = {'flags': 'sp'}

        if self.cfg.hasopt(sect, 'quad-pts'):
            kwargs['rule'] = self.cfg.get(sect, 'quad-pts')

        if self.cfg.hasopt(sect, 'quad-deg'):
            kwargs['qdeg'] = self.cfg.getint(sect, 'quad-deg')

        return get_quadrule(self.name, **kwargs)

    @property
    def qpts(self):
        return self._qrule.np_points

    @property
    def nqpts(self):
        return len(self.qpts)

    @property
    def qwts(self):
        return self._qrule.np_weights

    @lazyprop
    def fpts(self):
        rrule, ppts = {}, []

        for kind, proj, norm, area in self.faces:
            # Obtain the flux points in reference space for the face type
            try:
                r = rrule[kind]
            except KeyError:
                rule = self.cfg.get('solver-interfaces-' + kind, 'flux-pts')
                npts = self.npts_for_face[kind](self.order)

                rrule[kind] = r = get_quadrule(kind, rule, npts)

            # Project
            ppts.append(_proj_rule_pts(proj, r))

        return np.vstack(ppts)

    @lazyprop
    def fbasis_coeffs(self):
        coeffs = []
        rcache = {}

        # Suitable quadrature rules for various face types
        qrule_map = {
            'line': ('gauss-legendre', self.order + 1),
            'quad': ('gauss-legendre', (self.order + 1)**2),
            'tri': ('williams-shunn', 36)
        }

        for kind, proj, norm, area in self.faces:
            # Obtain a quadrature rule for integrating on the reference face
            # and evaluate this rule at the nodal basis defined by the flux
            # points
            try:
                qr, L = rcache[kind]
            except KeyError:
                qr = get_quadrule(kind, *qrule_map[kind])

                rule = self.cfg.get('solver-interfaces-' + kind, 'flux-pts')
                npts = self.npts_for_face[kind](self.order)

                pts = get_quadrule(kind, rule, npts).np_points
                ppb = get_polybasis(kind, self.order + 1, pts)

                L = ppb.nodal_basis_at(qr.np_points)

                rcache[kind] = (qr, L)

            # Do the quadrature
            M = self.ubasis.ortho_basis_at(_proj_rule_pts(proj, qr))
            S = np.einsum('i...,ik,ji->kj', area*qr.np_weights, L, M)

            coeffs.append(S)

        return np.vstack(coeffs)

    @chop
    def fbasis_at(self, pts):
        return np.dot(self.fbasis_coeffs, self.ubasis.ortho_basis_at(pts)).T

    @property
    def facenorms(self):
        return [norm for kind, proj, norm, area in self.faces]

    @lazyprop
    def norm_fpts(self):
        fnorms = self.facenorms
        return np.vstack([fn]*n for fn, n in zip(fnorms, self.nfacefpts))

    @lazyprop
    def spts(self):
        return self.std_ele(self.nsptsord - 1)

    @lazyprop
    def facefpts(self):
        nf = np.cumsum([0] + self.nfacefpts)
        return [list(xrange(nf[i], nf[i + 1])) for i in xrange(len(nf) - 1)]

    @lazyprop
    def nfacefpts(self):
        return [self.npts_for_face[kind](self.order)
                for kind, proj, norm, area in self.faces]

    @property
    def nfpts(self):
        return sum(self.nfacefpts)


class TensorProdShape(object):
    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)
        return list(p[::-1] for p in it.product(pts1d, repeat=cls.ndims))

    @property
    def nupts(self):
        return (self.order + 1)**self.ndims


class QuadShape(TensorProdShape, BaseShape):
    name = 'quad'
    ndims = 2

    # nspts = n^2
    nspts_coeffs = [1, 0, 0]
    nspts_cdenom = 1

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('line', lambda s: (s, -1), (0, -1), 1),
        ('line', lambda s: (1, s), (1, 0), 1),
        ('line', lambda s: (s, 1), (0, 1), 1),
        ('line', lambda s: (-1, s), (-1, 0), 1),
    ]


class HexShape(TensorProdShape, BaseShape):
    name = 'hex'
    ndims = 3

    # nspts = n^3
    nspts_coeffs = [1, 0, 0, 0]
    nspts_cdenom = 1

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('quad', lambda s, t: (s, t, -1), (0, 0, -1), 1),
        ('quad', lambda s, t: (s, -1, t), (0, -1, 0), 1),
        ('quad', lambda s, t: (1, s, t), (1, 0, 0), 1),
        ('quad', lambda s, t: (s, 1, t), (0, 1, 0), 1),
        ('quad', lambda s, t: (-1, s, t), (-1, 0, 0), 1),
        ('quad', lambda s, t: (s, t, 1), (0, 0, 1), 1),
    ]


class TriShape(BaseShape):
    name = 'tri'
    ndims = 2

    # nspts = n*(n + 1)/2
    nspts_coeffs = [1, 1, 0]
    nspts_cdenom = 2

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('line', lambda s: (s, -1), (0, -1), 1),
        ('line', lambda s: (-s, s), (1/sqrt(2), 1/sqrt(2)), sqrt(2)),
        ('line', lambda s: (-1, s), (-1, 0), 1),
    ]

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q)
                for i, q in enumerate(pts1d)
                for p in pts1d[:(sptord + 1 - i)]]

    @property
    def nupts(self):
        return (self.order + 1)*(self.order + 2) // 2


class TetShape(BaseShape):
    name = 'tet'
    ndims = 3

    # nspts = n*(n + 1)*(n + 2)/6
    nspts_coeffs = [1, 3, 2, 0]
    nspts_cdenom = 6

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('tri', lambda s, t: (s, t, -1), (0, 0, -1), 1),
        ('tri', lambda s, t: (s, -1, t), (0, -1, 0), 1),
        ('tri', lambda s, t: (-1, t, s), (-1, 0, 0), 1),
        ('tri', lambda s, t: (s, t, -s - t - 1),
         (1/sqrt(3), 1/sqrt(3), 1/sqrt(3)), sqrt(3)),
    ]

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q, r)
                for i, r in enumerate(pts1d)
                for j, q in enumerate(pts1d[:(sptord + 1 - i)])
                for p in pts1d[:(sptord + 1 - i - j)]]

    @property
    def nupts(self):
        return (self.order + 1)*(self.order + 2)*(self.order + 3) // 6


class PriShape(BaseShape):
    name = 'pri'
    ndims = 3

    # nspts = n^2*(n + 1)/2
    nspts_coeffs = [1, 1, 0, 0]
    nspts_cdenom = 2

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('tri', lambda s, t: (s, t, -1), (0, 0, -1), 1),
        ('tri', lambda s, t: (s, t, 1), (0, 0, 1), 1),
        ('quad', lambda s, t: (s, -1, t), (0, -1, 0), 1),
        ('quad', lambda s, t: (-s, s, t), (1/sqrt(2), 1/sqrt(2), 0), sqrt(2)),
        ('quad', lambda s, t: (-1, s, t), (-1, 0, 0), 1),
    ]

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q, r)
                for r in pts1d
                for i, q in enumerate(pts1d)
                for p in pts1d[:(sptord + 1 - i)]]

    @property
    def nupts(self):
        return (self.order + 1)**2*(self.order + 2) // 2
