# -*- coding: utf-8 -*-

from collections import Iterable
import functools as ft

from mpmath import mp
import numpy as np

from pyfr.mputil import jacobi, jacobian
from pyfr.nputil import chop
from pyfr.util import lazyprop


def get_polybasis(eletype, order, pts=[]):
    basis_map = {
        'line': _line_orthob_at,
        'quad': _quad_orthob_at,
        'tri': _tri_orthob_at,
        'tet': _tet_orthob_at,
        'pri': _pri_orthob_at,
        'hex': _hex_orthob_at
    }

    return PolyBasis(basis_map[eletype], order, pts)


class PolyBasis(object):
    def __init__(self, basis_at, order, pts):
        self._basis_at = ft.partial(basis_at, order)
        self._pts = pts

    @chop
    def ortho_basis_at(self, pts):
        if len(pts) and not isinstance(pts[0], Iterable):
            pts = [(p,) for p in pts]

        P = [self._basis_at(*p) for p in pts]
        P = np.array(P, dtype=np.float)

        return P.T

    @chop
    def jac_ortho_basis_at(self, pts):
        if len(pts) and not isinstance(pts[0], Iterable):
            pts = [(p,) for p in pts]

        J = [jacobian(self._basis_at, p) for p in pts]
        J = np.array(J, dtype=np.float)

        return J.swapaxes(0, 2)

    @chop
    def nodal_basis_at(self, epts):
        return np.dot(self.vdm_inv, self.ortho_basis_at(epts)).T

    @chop
    def jac_nodal_basis_at(self, epts):
        J = self.jac_ortho_basis_at(epts)

        return np.array([np.dot(self.vdm_inv, d) for d in J])

    @lazyprop
    def vdm(self):
        return self.ortho_basis_at(self._pts)

    @lazyprop
    def vdm_inv(self):
        return np.linalg.inv(self.vdm)


def _line_orthob_at(order, p):
    jp = jacobi(order - 1, 0, 0, p)
    return [mp.sqrt(i + 0.5)*p for i, p in enumerate(jp)]


def _tri_orthob_at(order, p, q):
    a = 2*(1 + p)/(1 - q) - 1 if q != 1 else 0
    b = q

    ob = []
    for i, pi in enumerate(jacobi(order - 1, 0, 0, a)):
        pa = pi*(1 - b)**i

        for j, pj in enumerate(jacobi(order - i - 1, 2*i + 1, 0, b)):
            cij = mp.sqrt((2*i + 1)*(2*i + 2*j + 2)) / 2**(i + 1)

            ob.append(cij*pa*pj)

    return ob


def _quad_orthob_at(order, p, q):
    sk = [mp.sqrt(k + 0.5) for k in xrange(order)]
    pa = [c*jp for c, jp in zip(sk, jacobi(order - 1, 0, 0, p))]
    pb = [c*jp for c, jp in zip(sk, jacobi(order - 1, 0, 0, q))]

    return [pi*pj for pi in pa for pj in pb]


def _tet_orthob_at(order, p, q, r):
    a = -2*(1 + p)/(q + r) - 1 if q + r != 0 else 0
    b = 2*(1 + q)/(1 - r) - 1 if r != 1 else 0
    c = r

    ob = []
    for i, pi in enumerate(jacobi(order - 1, 0, 0, a)):
        ci = mp.mpf(2)**(-2*i - 1.5)*mp.sqrt(4*i + 2)*(1 - b)**i

        for j, pj in enumerate(jacobi(order - i - 1, 2*i + 1, 0, b)):
            cj = mp.sqrt(i + j + 1)*2**-j*(1 - c)**(i + j)
            cij = ci*cj
            pij = pi*pj

            jp = jacobi(order - i - j - 1, 2*(i + j + 1), 0, c)
            for k, pk in enumerate(jp):
                ck = mp.sqrt(2*(k + j + i) + 3)

                ob.append(cij*ck*pij*pk)

    return ob


def _pri_orthob_at(order, p, q, r):
    a = 2*(1 + p)/(1 - q) - 1 if q != 1 else 0
    b = q
    c = r

    pab = []
    for i, pi in enumerate(jacobi(order - 1, 0, 0, a)):
        ci = (1 - b)**i / 2**(i + 1)

        for j, pj in enumerate(jacobi(order - i - 1, 2*i + 1, 0, b)):
            cij = mp.sqrt((2*i + 1)*(2*i + 2*j + 2))*ci

            pab.append(cij*pi*pj)

    sk = [mp.sqrt(k + 0.5) for k in xrange(order)]
    pc = [c*jp for c, jp in zip(sk, jacobi(order - 1, 0, 0, c))]

    return [pij*pk for pij in pab for pk in pc]


def _hex_orthob_at(order, p, q, r):
    sk = [mp.sqrt(k + 0.5) for k in xrange(order)]
    pa = [c*jp for c, jp in zip(sk, jacobi(order - 1, 0, 0, p))]
    pb = [c*jp for c, jp in zip(sk, jacobi(order - 1, 0, 0, q))]
    pc = [c*jp for c, jp in zip(sk, jacobi(order - 1, 0, 0, r))]

    return [pi*pj*pk for pi in pa for pj in pb for pk in pc]
