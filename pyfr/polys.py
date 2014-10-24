# -*- coding: utf-8 -*-

from collections import Iterable

from mpmath import mp
import numpy as np

from pyfr.mputil import jacobi, jacobian
from pyfr.nputil import chop
from pyfr.util import lazyprop, subclass_where


def get_polybasis(name, order, pts=[]):
    return subclass_where(BasePolyBasis, name=name)(order, pts)


class BasePolyBasis(object):
    name = None

    def __init__(self, order, pts):
        self.order = order
        self.pts = pts

    @chop
    def ortho_basis_at(self, pts):
        if len(pts) and not isinstance(pts[0], Iterable):
            pts = [(p,) for p in pts]

        P = [self.ortho_basis_at_mp(*p) for p in pts]
        P = np.array(P, dtype=np.float)

        return P.T

    @chop
    def jac_ortho_basis_at(self, pts):
        if len(pts) and not isinstance(pts[0], Iterable):
            pts = [(p,) for p in pts]

        J = [jacobian(self.ortho_basis_at_mp, p) for p in pts]
        J = np.array(J, dtype=np.float)

        return J.swapaxes(0, 2)

    @chop
    def nodal_basis_at(self, epts):
        return np.linalg.solve(self.vdm, self.ortho_basis_at(epts)).T

    @chop
    def jac_nodal_basis_at(self, epts):
        return np.linalg.solve(self.vdm, self.jac_ortho_basis_at(epts))

    @lazyprop
    def vdm(self):
        return self.ortho_basis_at(self.pts)


class LinePolyBasis(BasePolyBasis):
    name = 'line'

    def ortho_basis_at_mp(self, p):
        jp = jacobi(self.order - 1, 0, 0, p)
        return [mp.sqrt(i + 0.5)*p for i, p in enumerate(jp)]

    @lazyprop
    def degrees(self):
        return list(xrange(self.order))


class TriPolyBasis(BasePolyBasis):
    name = 'tri'

    def ortho_basis_at_mp(self, p, q):
        q = q if q != 1 else q + mp.eps

        a = 2*(1 + p)/(1 - q) - 1
        b = q

        ob = []
        for i, pi in enumerate(jacobi(self.order - 1, 0, 0, a)):
            pa = pi*(1 - b)**i

            for j, pj in enumerate(jacobi(self.order - i - 1, 2*i + 1, 0, b)):
                cij = mp.sqrt((2*i + 1)*(2*i + 2*j + 2)) / 2**(i + 1)

                ob.append(cij*pa*pj)

        return ob

    @lazyprop
    def degrees(self):
        return [i + j
                for i in xrange(self.order)
                for j in xrange(self.order - i)]


class QuadPolyBasis(BasePolyBasis):
    name = 'quad'

    def ortho_basis_at_mp(self, p, q):
        sk = [mp.sqrt(k + 0.5) for k in xrange(self.order)]
        pa = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, p))]
        pb = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, q))]

        return [pi*pj for pi in pa for pj in pb]

    @lazyprop
    def degrees(self):
        return [i + j for i in xrange(self.order) for j in xrange(self.order)]


class TetPolyBasis(BasePolyBasis):
    name = 'tet'

    def ortho_basis_at_mp(self, p, q, r):
        r = r if r != -q and r != 1 else r + mp.eps

        a = -2*(1 + p)/(q + r) - 1
        b = 2*(1 + q)/(1 - r) - 1
        c = r

        ob = []
        for i, pi in enumerate(jacobi(self.order - 1, 0, 0, a)):
            ci = mp.mpf(2)**(-2*i - 1.5)*mp.sqrt(4*i + 2)*(1 - b)**i

            for j, pj in enumerate(jacobi(self.order - i - 1, 2*i + 1, 0, b)):
                cj = mp.sqrt(i + j + 1)*2**-j*(1 - c)**(i + j)
                cij = ci*cj
                pij = pi*pj

                jp = jacobi(self.order - i - j - 1, 2*(i + j + 1), 0, c)
                for k, pk in enumerate(jp):
                    ck = mp.sqrt(2*(k + j + i) + 3)

                    ob.append(cij*ck*pij*pk)

        return ob

    @lazyprop
    def degrees(self):
        return [i + j + k
                for i in xrange(self.order)
                for j in xrange(self.order - i)
                for k in xrange(self.order - i - j)]


class PriPolyBasis(BasePolyBasis):
    name = 'pri'

    def ortho_basis_at_mp(self, p, q, r):
        q = q if q != 1 else q + mp.eps

        a = 2*(1 + p)/(1 - q) - 1
        b = q
        c = r

        pab = []
        for i, pi in enumerate(jacobi(self.order - 1, 0, 0, a)):
            ci = (1 - b)**i / 2**(i + 1)

            for j, pj in enumerate(jacobi(self.order - i - 1, 2*i + 1, 0, b)):
                cij = mp.sqrt((2*i + 1)*(2*i + 2*j + 2))*ci

                pab.append(cij*pi*pj)

        sk = [mp.sqrt(k + 0.5) for k in xrange(self.order)]
        pc = [s*jp for s, jp in zip(sk, jacobi(self.order - 1, 0, 0, c))]

        return [pij*pk for pij in pab for pk in pc]

    @lazyprop
    def degrees(self):
        return [i + j + k
                for i in xrange(self.order)
                for j in xrange(self.order - i)
                for k in xrange(self.order)]


class HexPolyBasis(BasePolyBasis):
    name = 'hex'

    def ortho_basis_at_mp(self, p, q, r):
        sk = [mp.sqrt(k + 0.5) for k in xrange(self.order)]
        pa = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, p))]
        pb = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, q))]
        pc = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, r))]

        return [pi*pj*pk for pi in pa for pj in pb for pk in pc]

    @lazyprop
    def degrees(self):
        return [i + j + k
                for i in xrange(self.order)
                for j in xrange(self.order)
                for k in xrange(self.order)]
