# -*- coding: utf-8 -*-

import itertools as it
from math import exp
import re

import numpy as np

from pyfr.nputil import block_diag, clean
from pyfr.polys import get_polybasis
from pyfr.quadrules import get_quadrule
from pyfr.util import lazyprop


def _proj_pts(projector, pts):
    pts = np.atleast_2d(pts.T)
    return np.vstack(np.broadcast_arrays(*projector(*pts))).T


@clean
def _proj_l2(qrule, basis):
    return basis.vdm.T @ (qrule.wts*basis.ortho_basis_at(qrule.pts))


class BaseShape(object):
    name = None
    ndims = -1

    npts_coeffs = None
    npts_cdenom = None

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
        self.antialias = {s.strip() for s in self.antialias.split(',')}
        self.antialias.discard('none')
        if self.antialias - {'flux', 'div-flux', 'surf-flux'}:
            raise ValueError('Invalid anti-alias options')

        self.ubasis = get_polybasis(self.name, self.order + 1, self.upts)

        if nspts:
            self.nsptsord = nsptord = self.order_from_nspts(nspts)
            self.sbasis = get_polybasis(self.name, nsptord, self.spts)

            # Basis for free-stream metric
            # We need p-th order pseudo grid points, which includes
            # p-th order points on faces.
            # It guarantees th q-th order collocation projection on the face
            # on the both adjacent cells.
            # Ref. 1 JCP 281, 28-54, Sec 4.2
            # Ref. 2 JSC 26(3), 301-327, Definition 1
            self.mbasis = get_polybasis(self.name, max(self.order + 1, 2),
                                        self.mpts)

    @classmethod
    def nspts_from_order(cls, sptord):
        return np.polyval(cls.npts_coeffs, sptord) // cls.npts_cdenom

    @classmethod
    def order_from_nspts(cls, nspts):
        # Obtain the coefficients for the poly: P(n) - nspts = 0
        coeffs = list(cls.npts_coeffs)
        coeffs[-1] -= cls.npts_cdenom*int(nspts)

        # Iterate
        for n in range(1, 15):
            if np.polyval(coeffs, n) == 0:
                return n
        else:
            raise ValueError('Invalid number of shape points')

    @clean
    def opmat(self, expr):
        expr = expr.lower().replace('*', '@')

        if not re.match(r'[m0-9\-+@() ]+$', expr):
            raise ValueError('Invalid operator matrix expression')

        mats = {m: getattr(self, m) for m in re.findall(r'm\d+', expr)}
        return eval(expr, {'__builtins__': None}, mats)

    @lazyprop
    def m0(self):
        return self.ubasis.nodal_basis_at(self.fpts)

    @lazyprop
    def m1(self):
        m = np.rollaxis(self.ubasis.jac_nodal_basis_at(self.upts), 2)
        return m.reshape(self.nupts, -1)

    @lazyprop
    def m2(self):
        m = self.norm_fpts[..., None]*self.m0[:, None, :]
        return m.reshape(self.nfpts, -1)

    @lazyprop
    def m3(self):
        m = self.gbasis_at(self.upts)

        if 'surf-flux' in self.antialias:
            fp = [_proj_l2(self._iqrules[kind], self.facebases[kind])
                  for kind, proj, norm in self.faces]

            m = m @ block_diag(fp)

        return m

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
    def m9(self):
        return _proj_l2(self._eqrule, self.ubasis)

    @property
    def m10(self):
        return block_diag([self.m9]*self.ndims)

    @lazyprop
    @clean
    def m11(self):
        ub = self.ubasis

        n = max(sum(dd) for dd in ub.degrees)
        ncut = self.cfg.getint('soln-filter', 'cutoff')
        order = self.cfg.getint('soln-filter', 'order')
        alpha = self.cfg.getfloat('soln-filter', 'alpha')

        A = np.ones(self.nupts)
        for i, d in enumerate(sum(dd) for dd in ub.degrees):
            if d >= ncut < n:
                A[i] = exp(-alpha*((d - ncut)/(n - ncut))**order)

        return np.linalg.solve(ub.vdm, A[:, None]*ub.vdm).T

    @lazyprop
    def nupts(self):
        n = self.order + 1
        return np.polyval(self.npts_coeffs, n) // self.npts_cdenom

    @lazyprop
    def upts(self):
        rname = self.cfg.get('solver-elements-' + self.name, 'soln-pts')
        return get_quadrule(self.name, rname, self.nupts).pts

    def _get_qrule(self, eleint, kind, **kwargs):
        sect = 'solver-{0}-{1}'.format(eleint, kind)

        if self.cfg.hasopt(sect, 'quad-pts'):
            kwargs['rule'] = self.cfg.get(sect, 'quad-pts')

        if self.cfg.hasopt(sect, 'quad-deg'):
            kwargs['qdeg'] = self.cfg.getint(sect, 'quad-deg')

        return get_quadrule(kind, **kwargs)

    @lazyprop
    def _eqrule(self):
        return self._get_qrule('elements', self.name)

    @lazyprop
    def _iqrules(self):
        return {kind: self._get_qrule('interfaces', kind, flags='s')
                for kind in {k for k, p, n in self.faces}}

    @property
    def qpts(self):
        return self._eqrule.pts

    @property
    def nqpts(self):
        return len(self.qpts)

    @lazyprop
    def fpts(self):
        ppts = []

        for kind, proj, norm in self.faces:
            # Obtain the flux points in reference space for the face type
            if 'surf-flux' in self.antialias:
                r = self._iqrules[kind]
            else:
                rule = self.cfg.get('solver-interfaces-' + kind, 'flux-pts')
                npts = self.npts_for_face[kind](self.order)

                r = get_quadrule(kind, rule, npts)

            # Project
            ppts.append(_proj_pts(proj, r.pts))

        return np.vstack(ppts)

    @lazyprop
    def fpts_wts(self):
        pwts = []

        for kind, proj, norm in self.faces:
            # Obtain the weights in reference space for the face type
            if 'surf-flux' in self.antialias:
                r = self._iqrules[kind]
            else:
                rule = self.cfg.get('solver-interfaces-' + kind, 'flux-pts')
                npts = self.npts_for_face[kind](self.order)

                r = get_quadrule(kind, rule, npts)

            pwts.append(r.wts)

        return np.hstack(pwts)

    @lazyprop
    def gbasis_coeffs(self):
        coeffs = []

        # Suitable quadrature rules for various face types
        qrule_map = {
            'line': ('gauss-legendre', self.order + 1),
            'quad': ('gauss-legendre', (self.order + 1)**2),
            'tri': ('williams-shunn', 36)
        }

        for kind, proj, norm in self.faces:
            # Obtain a quadrature rule for integrating on the reference face
            # and evaluate this rule at the nodal basis defined by the flux
            # points
            qr = get_quadrule(kind, *qrule_map[kind])
            L = self.facebases[kind].nodal_basis_at(qr.pts)

            # Do the quadrature
            M = self.ubasis.ortho_basis_at(_proj_pts(proj, qr.pts))
            S = np.einsum('i...,ik,ji->kj', qr.wts, L, M)

            coeffs.append(S)

        return np.vstack(coeffs)

    @clean
    def gbasis_at(self, pts):
        return (self.gbasis_coeffs @ self.ubasis.ortho_basis_at(pts)).T

    @property
    def facenorms(self):
        return [norm for kind, proj, norm in self.faces]

    @lazyprop
    def norm_fpts(self):
        fnorms = self.facenorms
        return np.vstack([[fn]*n for fn, n in zip(fnorms, self.nfacefpts)])

    @lazyprop
    def spts(self):
        return self.std_ele(self.nsptsord - 1)

    @lazyprop
    def facebases(self):
        fb = {}

        for kind in {k for k, p, n in self.faces}:
            rule = self.cfg.get('solver-interfaces-' + kind, 'flux-pts')
            npts = self.npts_for_face[kind](self.order)

            pts = get_quadrule(kind, rule, npts).pts

            fb[kind] = get_polybasis(kind, self.order + 1, pts)

        return fb

    @lazyprop
    def facefpts(self):
        nf = np.cumsum([0] + self.nfacefpts)
        return [list(range(nf[i], nf[i + 1])) for i in range(len(nf) - 1)]

    @lazyprop
    def nfacefpts(self):
        if 'surf-flux' in self.antialias:
            cnt = lambda k: len(self._iqrules[k].pts)
        else:
            cnt = lambda k: self.npts_for_face[k](self.order)

        return [cnt(kind) for kind, proj, norm in self.faces]

    @property
    def nfpts(self):
        return sum(self.nfacefpts)

    @lazyprop
    def mpts(self):
        return self.std_ele(max(self.order, 1))

    @lazyprop
    def nmpts(self):
        return len(self.mpts)


class TensorProdShape(object):
    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)
        return [p[::-1] for p in it.product(pts1d, repeat=cls.ndims)]


class QuadShape(TensorProdShape, BaseShape):
    name = 'quad'
    ndims = 2

    # nspts = n^2
    npts_coeffs = [1, 0, 0]
    npts_cdenom = 1

    # Faces: type, reference-to-face projection, normal
    faces = [
        ('line', lambda s: (s, -1), (0, -1)),
        ('line', lambda s: (1, s), (1, 0)),
        ('line', lambda s: (s, 1), (0, 1)),
        ('line', lambda s: (-1, s), (-1, 0)),
    ]


class HexShape(TensorProdShape, BaseShape):
    name = 'hex'
    ndims = 3

    # nspts = n^3
    npts_coeffs = [1, 0, 0, 0]
    npts_cdenom = 1

    # Faces: type, reference-to-face projection, normal
    faces = [
        ('quad', lambda s, t: (s, t, -1), (0, 0, -1)),
        ('quad', lambda s, t: (s, -1, t), (0, -1, 0)),
        ('quad', lambda s, t: (1, s, t), (1, 0, 0)),
        ('quad', lambda s, t: (s, 1, t), (0, 1, 0)),
        ('quad', lambda s, t: (-1, s, t), (-1, 0, 0)),
        ('quad', lambda s, t: (s, t, 1), (0, 0, 1)),
    ]


class TriShape(BaseShape):
    name = 'tri'
    ndims = 2

    # nspts = n*(n + 1)/2
    npts_coeffs = [1, 1, 0]
    npts_cdenom = 2

    # Faces: type, reference-to-face projection, normal
    faces = [
        ('line', lambda s: (s, -1), (0, -1)),
        ('line', lambda s: (-s, s), (1, 1)),
        ('line', lambda s: (-1, s), (-1, 0)),
    ]

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q)
                for i, q in enumerate(pts1d)
                for p in pts1d[:(sptord + 1 - i)]]


class TetShape(BaseShape):
    name = 'tet'
    ndims = 3

    # nspts = n*(n + 1)*(n + 2)/6
    npts_coeffs = [1, 3, 2, 0]
    npts_cdenom = 6

    # Faces: type, reference-to-face projection, normal
    faces = [
        ('tri', lambda s, t: (s, t, -1), (0, 0, -1)),
        ('tri', lambda s, t: (s, -1, t), (0, -1, 0)),
        ('tri', lambda s, t: (-1, t, s), (-1, 0, 0)),
        ('tri', lambda s, t: (s, t, -s - t - 1), (1, 1, 1)),
    ]

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q, r)
                for i, r in enumerate(pts1d)
                for j, q in enumerate(pts1d[:(sptord + 1 - i)])
                for p in pts1d[:(sptord + 1 - i - j)]]


class PriShape(BaseShape):
    name = 'pri'
    ndims = 3

    # nspts = n^2*(n + 1)/2
    npts_coeffs = [1, 1, 0, 0]
    npts_cdenom = 2

    # Faces: type, reference-to-face projection, normal
    faces = [
        ('tri', lambda s, t: (s, t, -1), (0, 0, -1)),
        ('tri', lambda s, t: (s, t, 1), (0, 0, 1)),
        ('quad', lambda s, t: (s, -1, t), (0, -1, 0)),
        ('quad', lambda s, t: (-s, s, t), (1, 1, 0)),
        ('quad', lambda s, t: (-1, s, t), (-1, 0, 0)),
    ]

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q, r)
                for r in pts1d
                for i, q in enumerate(pts1d)
                for p in pts1d[:(sptord + 1 - i)]]


class PyrShape(BaseShape):
    name = 'pyr'
    ndims = 3

    # nspts = n*(n + 1)*(2*n + 1)/6
    npts_coeffs = [2, 3, 1, 0]
    npts_cdenom = 6

    # Faces: type, reference-to-face projection, normal
    faces = [
        ('quad', lambda s, t: (s, t, -1), (0, 0, -1)),
        ('tri', lambda s, t: (s + (t + 1)/2, (t - 1)/2, t), (0, -1, 0.5)),
        ('tri', lambda s, t: ((1 - t)/2, -s - (t + 1)/2, t), (1, 0, 0.5)),
        ('tri', lambda s, t: (-s - (t + 1)/2, (1 - t)/2, t), (0, 1, 0.5)),
        ('tri', lambda s, t: ((t - 1)/2, s + (t + 1)/2, t), (-1, 0, 0.5)),
    ]

    @classmethod
    def std_ele(cls, sptord):
        npts1d = 2*sptord + 1
        pts1d = np.linspace(-1, 1, npts1d)

        return [(p, q, r)
                for i, r in enumerate(pts1d[::2])
                for q in pts1d[i:npts1d - i:2]
                for p in pts1d[i:npts1d - i:2]]
