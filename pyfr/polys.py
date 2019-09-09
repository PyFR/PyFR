# -*- coding: utf-8 -*-

from collections import Iterable
from math import sqrt

import numpy as np

from pyfr.nputil import clean
from pyfr.util import lazyprop, subclass_where


def jacobi(n, a, b, z):
    j = [1]

    if n >= 1:
        j.append(((a + b + 2)*z + a - b) / 2)
    if n >= 2:
        apb, bbmaa = a + b, b*b - a*a

        for q in range(2, n + 1):
            qapbpq, apbp2q = q*(apb + q), apb + 2*q
            apbp2qm1, apbp2qm2 = apbp2q - 1, apbp2q - 2

            aq = apbp2q*apbp2qm1/(2*qapbpq)
            bq = apbp2qm1*bbmaa/(2*qapbpq*apbp2qm2)
            cq = apbp2q*(a + q - 1)*(b + q - 1)/(qapbpq*apbp2qm2)

            # Update
            j.append((aq*z - bq)*j[-1] - cq*j[-2])

    return j


def jacobi_diff(n, a, b, z):
    dj = [0]

    if n >= 1:
        dj.extend(jp*(i + a + b + 2)/2
                  for i, jp in enumerate(jacobi(n - 1, a + 1, b + 1, z)))

    return dj


def get_polybasis(name, order, pts=[]):
    return subclass_where(BasePolyBasis, name=name)(order, pts)


class BasePolyBasis(object):
    name = None

    def __init__(self, order, pts):
        self.order = order
        self.pts = pts

    @clean
    def ortho_basis_at(self, pts):
        if len(pts) and not isinstance(pts[0], Iterable):
            pts = [(p,) for p in pts]

        return np.array([self.ortho_basis_at_py(*p) for p in pts]).T

    @clean
    def jac_ortho_basis_at(self, pts):
        if len(pts) and not isinstance(pts[0], Iterable):
            pts = [(p,) for p in pts]

        J = [self.jac_ortho_basis_at_py(*p) for p in pts]

        return np.array(J).swapaxes(0, 2)

    @clean
    def nodal_basis_at(self, epts):
        return np.linalg.solve(self.vdm, self.ortho_basis_at(epts)).T

    @clean
    def jac_nodal_basis_at(self, epts):
        return np.linalg.solve(self.vdm, self.jac_ortho_basis_at(epts))

    @lazyprop
    def vdm(self):
        return self.ortho_basis_at(self.pts)

    def proj_to(self, tobasis):
        if tobasis.order > self.order:
            return self.nodal_basis_at(tobasis.pts)
        elif tobasis.order < self.order:
            degmap = {dd: i for i, dd in enumerate(self.degrees)}
            todegs = [degmap[dd] for dd in tobasis.degrees]

            return tobasis.vdm.T @ self.invvdm.T[todegs]
        else:
            return np.eye(len(self.pts))

    @lazyprop
    @clean
    def invvdm(self):
        return np.linalg.inv(self.vdm)


class LinePolyBasis(BasePolyBasis):
    name = 'line'

    def ortho_basis_at_py(self, p):
        jp = jacobi(self.order - 1, 0, 0, p)
        return [sqrt(i + 0.5)*p for i, p in enumerate(jp)]

    def jac_ortho_basis_at_py(self, p):
        djp = jacobi_diff(self.order - 1, 0, 0, p)
        return [(sqrt(i + 0.5)*p,) for i, p in enumerate(djp)]

    @lazyprop
    def degrees(self):
        return [(i,) for i in range(self.order)]


class TriPolyBasis(BasePolyBasis):
    name = 'tri'

    def ortho_basis_at_py(self, p, q):
        a = 2*(1 + p)/(1 - q) - 1 if q != 1 else -1
        b = q

        ob = []
        for i, pi in enumerate(jacobi(self.order - 1, 0, 0, a)):
            pa = pi*(1 - b)**i

            for j, pj in enumerate(jacobi(self.order - i - 1, 2*i + 1, 0, b)):
                cij = sqrt((2*i + 1)*(2*i + 2*j + 2)) / 2**(i + 1)

                ob.append(cij*pa*pj)

        return ob

    def jac_ortho_basis_at_py(self, p, q):
        a = 2*(1 + p)/(1 - q) - 1 if q != 1 else -1
        b = q

        f = jacobi(self.order - 1, 0, 0, a)
        df = jacobi_diff(self.order - 1, 0, 0, a)

        ob = []
        for i, (fi, dfi) in enumerate(zip(f, df)):
            g = jacobi(self.order - i - 1, 2*i + 1, 0, b)
            dg = jacobi_diff(self.order - i - 1, 2*i + 1, 0, b)

            for j, (gj, dgj) in enumerate(zip(g, dg)):
                cij = sqrt((2*i + 1)*(2*i + 2*j + 2)) / 2**(i + 1)

                tmp = (1 - b)**(i - 1) if i > 0 else 1

                pij = 2*tmp*dfi*gj
                qij = tmp*(-i*fi + (1 + a)*dfi)*gj + (1 - b)**i*fi*dgj

                ob.append([cij*pij, cij*qij])

        return ob

    @lazyprop
    def degrees(self):
        return [(i, j)
                for i in range(self.order)
                for j in range(self.order - i)]


class QuadPolyBasis(BasePolyBasis):
    name = 'quad'

    def ortho_basis_at_py(self, p, q):
        sk = [sqrt(k + 0.5) for k in range(self.order)]
        pa = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, p))]
        pb = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, q))]

        return [pi*pj for pi in pa for pj in pb]

    def jac_ortho_basis_at_py(self, p, q):
        sk = [sqrt(k + 0.5) for k in range(self.order)]
        pa = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, p))]
        pb = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, q))]

        dpa = [c*jp for c, jp in zip(sk, jacobi_diff(self.order - 1, 0, 0, p))]
        dpb = [c*jp for c, jp in zip(sk, jacobi_diff(self.order - 1, 0, 0, q))]

        return [[dpi*pj, pi*dpj]
                for pi, dpi in zip(pa, dpa)
                for pj, dpj in zip(pb, dpb)]

    @lazyprop
    def degrees(self):
        return [(i, j) for i in range(self.order) for j in range(self.order)]


class TetPolyBasis(BasePolyBasis):
    name = 'tet'

    def ortho_basis_at_py(self, p, q, r):
        a = -2*(1 + p)/(q + r) - 1 if r != -q else -1
        b = 2*(1 + q)/(1 - r) - 1 if r != 1 else -1
        c = r

        ob = []
        for i, pi in enumerate(jacobi(self.order - 1, 0, 0, a)):
            ci = 2**(-2*i - 1)*sqrt(2*i + 1)*(1 - b)**i

            for j, pj in enumerate(jacobi(self.order - i - 1, 2*i + 1, 0, b)):
                cj = sqrt(i + j + 1)*2**-j*(1 - c)**(i + j)
                cij = ci*cj
                pij = pi*pj

                jp = jacobi(self.order - i - j - 1, 2*(i + j + 1), 0, c)
                for k, pk in enumerate(jp):
                    ck = sqrt(2*(k + j + i) + 3)

                    ob.append(cij*ck*pij*pk)

        return ob

    def jac_ortho_basis_at_py(self, p, q, r):
        a = -2*(1 + p)/(q + r) - 1 if r != -q else -1
        b = 2*(1 + q)/(1 - r) - 1 if r != 1 else -1
        c = r

        f = jacobi(self.order - 1, 0, 0, a)
        df = jacobi_diff(self.order - 1, 0, 0, a)

        ob = []
        for i, (fi, dfi) in enumerate(zip(f, df)):
            ci = 2**(-2*i - 1)*sqrt(2*i + 1)
            g = jacobi(self.order - i - 1, 2*i + 1, 0, b)
            dg = jacobi_diff(self.order - i - 1, 2*i + 1, 0, b)

            for j, (gj, dgj) in enumerate(zip(g, dg)):
                cj = sqrt(i + j + 1)*2**-j
                cij = ci*cj
                h = jacobi(self.order - i - j - 1, 2*(i + j + 1), 0, c)
                dh = jacobi_diff(self.order - i - j - 1, 2*(i + j + 1), 0, c)

                for k, (hk, dhk) in enumerate(zip(h, dh)):
                    ck = sqrt(2*(k + j + i) + 3)
                    cijk = cij*ck

                    tmp1 = (1 - c)**(i + j - 1) if i + j > 0 else 1
                    tmp2 = tmp1*(1 - b)**(i - 1) if i > 0 else 1

                    pijk = 4*tmp2*dfi*gj*hk
                    qijk = 2*(tmp2*(-i*fi + (1 + a)*dfi)*gj
                              + tmp1*(1 - b)**i*fi*dgj)*hk

                    rijk = (
                        2*(1 + a)*tmp2*dfi*gj*hk
                        + (1 + b)*tmp1*(1 - b)**i*fi*dgj*hk
                        + (1 - c)**(i + j)*(1 - b)**i*fi*gj*dhk
                        - (i*(1 + b)*tmp2 + (i + j)*tmp1*(1 - b)**i)*fi*gj*hk
                    )

                    ob.append([cijk*pijk, cijk*qijk, cijk*rijk])

        return ob

    @lazyprop
    def degrees(self):
        return [(i, j, k)
                for i in range(self.order)
                for j in range(self.order - i)
                for k in range(self.order - i - j)]


class PriPolyBasis(BasePolyBasis):
    name = 'pri'

    def ortho_basis_at_py(self, p, q, r):
        a = 2*(1 + p)/(1 - q) - 1 if q != 1 else -1
        b = q
        c = r

        pab = []
        for i, pi in enumerate(jacobi(self.order - 1, 0, 0, a)):
            ci = (1 - b)**i / 2**(i + 1)

            for j, pj in enumerate(jacobi(self.order - i - 1, 2*i + 1, 0, b)):
                cij = sqrt((2*i + 1)*(2*i + 2*j + 2))*ci

                pab.append(cij*pi*pj)

        sk = [sqrt(k + 0.5) for k in range(self.order)]
        pc = [s*jp for s, jp in zip(sk, jacobi(self.order - 1, 0, 0, c))]

        return [pij*pk for pij in pab for pk in pc]

    def jac_ortho_basis_at_py(self, p, q, r):
        a = 2*(1 + p)/(1 - q) - 1 if q != 1 else -1
        b = q
        c = r

        f = jacobi(self.order - 1, 0, 0, a)
        df = jacobi_diff(self.order - 1, 0, 0, a)

        pab = []
        for i, (fi, dfi) in enumerate(zip(f, df)):
            g = jacobi(self.order - i - 1, 2*i + 1, 0, b)
            dg = jacobi_diff(self.order - i - 1, 2*i + 1, 0, b)

            for j, (gj, dgj) in enumerate(zip(g, dg)):
                cij = sqrt((2*i + 1)*(2*i + 2*j + 2)) / 2**(i + 1)

                tmp = (1 - b)**(i - 1) if i > 0 else 1

                pij = 2*tmp*dfi*gj
                qij = tmp*(-i*fi + (1 + a)*dfi)*gj + (1 - b)**i*fi*dgj
                rij = (1 - b)**i*fi*gj

                pab.append([cij*pij, cij*qij, cij*rij])

        sk = [sqrt(k + 0.5) for k in range(self.order)]
        hc = [s*jp for s, jp in zip(sk, jacobi(self.order - 1, 0, 0, c))]
        dhc = [s*jp for s, jp in zip(sk, jacobi_diff(self.order - 1, 0, 0, c))]

        return [[pij*hk, qij*hk, rij*dhk]
                for pij, qij, rij in pab for hk, dhk in zip(hc, dhc)]

    @lazyprop
    def degrees(self):
        return [(i, j, k)
                for i in range(self.order)
                for j in range(self.order - i)
                for k in range(self.order)]


class PyrPolyBasis(BasePolyBasis):
    name = 'pyr'

    def ortho_basis_at_py(self, p, q, r):
        a = 2*p/(1 - r) if r != 1 else 0
        b = 2*q/(1 - r) if r != 1 else 0
        c = r

        sk = [2**(-k - 0.25)*sqrt(k + 0.5)
              for k in range(self.order)]
        pa = [s*jp for s, jp in zip(sk, jacobi(self.order - 1, 0, 0, a))]
        pb = [s*jp for s, jp in zip(sk, jacobi(self.order - 1, 0, 0, b))]

        ob = []
        for i, pi in enumerate(pa):
            for j, pj in enumerate(pb):
                cij = (1 - c)**(i + j)
                pij = pi*pj

                pc = jacobi(self.order - max(i, j) - 1, 2*(i + j + 1), 0, c)
                for k, pk in enumerate(pc):
                    ck = sqrt(2*(k + j + i) + 3)

                    ob.append(cij*ck*pij*pk)

        return ob

    def jac_ortho_basis_at_py(self, p, q, r):
        a = 2*p/(1 - r) if r != 1 else 0
        b = 2*q/(1 - r) if r != 1 else 0
        c = r

        sk = [2**(-k - 0.25)*sqrt(k + 0.5)
              for k in range(self.order)]
        fc = [s*jp for s, jp in zip(sk, jacobi(self.order - 1, 0, 0, a))]
        gc = [s*jp for s, jp in zip(sk, jacobi(self.order - 1, 0, 0, b))]

        dfc = [s*jp for s, jp in zip(sk, jacobi_diff(self.order - 1, 0, 0, a))]
        dgc = [s*jp for s, jp in zip(sk, jacobi_diff(self.order - 1, 0, 0, b))]

        ob = []
        for i, (fi, dfi) in enumerate(zip(fc, dfc)):
            for j, (gj, dgj) in enumerate(zip(gc, dgc)):
                h = jacobi(self.order - max(i, j) - 1, 2*(i + j + 1), 0, c)
                dh = jacobi_diff(
                    self.order - max(i, j) - 1, 2*(i + j + 1), 0, c
                )

                for k, (hk, dhk) in enumerate(zip(h, dh)):
                    ck = sqrt(2*(k + j + i) + 3)

                    tmp = (1 - c)**(i + j-1) if i + j > 0 else 1

                    pijk = 2*tmp*dfi*gj*hk
                    qijk = 2*tmp*fi*dgj*hk
                    rijk = (tmp*(a*dfi*gj + b*fi*dgj - (i + j)*fi*gj)*hk
                            + (1 - c)**(i + j)*fi*gj*dhk)

                    ob.append([ck*pijk, ck*qijk, ck*rijk])

        return ob

    @lazyprop
    def degrees(self):
        return [(i, j, k)
                for i in range(self.order)
                for j in range(self.order)
                for k in range(self.order - max(i, j))]


class HexPolyBasis(BasePolyBasis):
    name = 'hex'

    def ortho_basis_at_py(self, p, q, r):
        sk = [sqrt(k + 0.5) for k in range(self.order)]
        pa = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, p))]
        pb = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, q))]
        pc = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, r))]

        return [pi*pj*pk for pi in pa for pj in pb for pk in pc]

    def jac_ortho_basis_at_py(self, p, q, r):
        sk = [sqrt(k + 0.5) for k in range(self.order)]
        pa = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, p))]
        pb = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, q))]
        pc = [c*jp for c, jp in zip(sk, jacobi(self.order - 1, 0, 0, r))]

        dpa = [c*jp for c, jp in zip(sk, jacobi_diff(self.order - 1, 0, 0, p))]
        dpb = [c*jp for c, jp in zip(sk, jacobi_diff(self.order - 1, 0, 0, q))]
        dpc = [c*jp for c, jp in zip(sk, jacobi_diff(self.order - 1, 0, 0, r))]

        return [[dpi*pj*pk, pi*dpj*pk, pi*pj*dpk]
                for pi, dpi in zip(pa, dpa)
                for pj, dpj in zip(pb, dpb)
                for pk, dpk in zip(pc, dpc)]

    @lazyprop
    def degrees(self):
        return [(i, j, k)
                for i in range(self.order)
                for j in range(self.order)
                for k in range(self.order)]
