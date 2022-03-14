# -*- coding: utf-8 -*-

import ctypes as ct
import enum
import functools as ft
import itertools as it
import re

import numpy as np
from pyfr.util import memoize


def block_diag(arrs):
    shapes = [a.shape for a in arrs]
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc

    return out


def clean(origfn=None, tol=1e-10):
    def cleanfn(fn):
        @ft.wraps(fn)
        def newfn(*args, **kwargs):
            arr = fn(*args, **kwargs).copy()

            # Flush small elements to zero
            arr[np.abs(arr) < tol] = 0

            # Coalesce similar elements
            if arr.size > 1:
                amfl = np.abs(arr.flat)
                amix = np.argsort(amfl)

                i, ix = 0, amix[0]
                for j, jx in enumerate(amix[1:], start=1):
                    if amfl[jx] - amfl[ix] >= tol:
                        if j - i > 1:
                            amfl[amix[i:j]] = np.median(amfl[amix[i:j]])
                        i, ix = j, jx

                if i != j:
                    amfl[amix[i:]] = np.median(amfl[amix[i:]])

                # Fix up the signs and assign
                arr.flat = np.copysign(amfl, arr.flat)

            return arr
        return newfn

    return cleanfn(origfn) if origfn else cleanfn


_npeval_syms = {
    '__builtins__': None,
    'exp': np.exp, 'log': np.log,
    'sin': np.sin, 'asin': np.arcsin,
    'cos': np.cos, 'acos': np.arccos,
    'tan': np.tan, 'atan': np.arctan, 'atan2': np.arctan2,
    'abs': np.abs, 'pow': np.power, 'sqrt': np.sqrt,
    'tanh': np.tanh, 'pi': np.pi,
    'max': np.maximum, 'min': np.minimum
}


def npeval(expr, locals):
    # Disallow direct exponentiation
    if '^' in expr or '**' in expr:
        raise ValueError('Direct exponentiation is not supported; use pow')

    # Ensure the expression does not contain invalid characters
    if not re.match(r'[A-Za-z0-9_ \t\n\r.,+\-*/%()]+$', expr):
        raise ValueError('Invalid characters in expression')

    # Disallow access to object attributes
    objs = '|'.join(it.chain(_npeval_syms, locals))
    if re.search(rf'({objs}|\))\s*\.', expr):
        raise ValueError('Invalid expression')

    return eval(expr, _npeval_syms, locals)


def fuzzysort(arr, idx, dim=0, tol=1e-6):
    # Extract our dimension and argsort
    arrd = arr[dim]
    srtdidx = sorted(idx, key=arrd.__getitem__)

    if len(srtdidx) > 1:
        i, ix = 0, srtdidx[0]
        for j, jx in enumerate(srtdidx[1:], start=1):
            if arrd[jx] - arrd[ix] >= tol:
                if j - i > 1:
                    srtdidx[i:j] = fuzzysort(arr, srtdidx[i:j], dim + 1, tol)
                i, ix = j, jx

        if i != j:
            srtdidx[i:] = fuzzysort(arr, srtdidx[i:], dim + 1, tol)

    return srtdidx


_ctype_map = {
    np.int32: 'int', np.uint32: 'unsigned int',
    np.int64: 'long long', np.uint64: 'unsigned long long',
    np.float32: 'float', np.float64: 'double'
}


def npdtype_to_ctype(dtype):
    return _ctype_map[np.dtype(dtype).type]


_ctypestype_map = {
    np.int32: ct.c_int32, np.uint32: ct.c_uint32,
    np.int64: ct.c_int64, np.uint64: ct.c_uint64,
    np.float32: ct.c_float, np.float64: ct.c_double
}


def npdtype_to_ctypestype(dtype):
    # Special-case None which otherwise expands to np.float
    if dtype is None:
        return None

    return _ctypestype_map[np.dtype(dtype).type]


#Get indices for interval segments of interpolation
def get_interval_idxs(t, xarr):
    if np.asarray(xarr).size > 1:
        idxs = np.array([np.max(np.nonzero(xx >= t[:-1] - 1e-12)[0]) 
                                                    for xx in xarr])
    else:
        idxs = np.max(np.nonzero(xarr >= t[:-1] - 1e-12)[0])
    return idxs


#Modified Gauss Elimination/Thompson algorithm for tridiag systems
#All changes are done in place and the algorithm 
#is applied to the lhs and rhs together
def trislv(abc, d):
    a, b, c = abc
    #forward elimination
    d[0] /= b[0]
    for i, (ai, bi, bj, ci, dj) in enumerate(
                zip(a, b[1: ], b[: -1], c, d[: -1]), start=1):
        #lhs coeffs
        c[i-1] = cdivb = ci/bj
        b[i] = bi1 = bi - ai*cdivb
        #rhs     
        d[i] -= ai*dj
        d[i] /= bi1

    #backward substitution
    dd = d[::-1]
    for i, (ci, dj) in enumerate(zip(c[::-1], d[:0:-1]), start=1):
        dd[i] -= ci*dj

#tridiagonal solver for cyclic/periodic systems 
#Using Sherman-Morrison formula and Thompson algorithm
# A = A + u ⊗ v
# u, v are vectors
def trislvcyc(abc, d, alpha, beta):
    b = abc[1]
    ud = np.hstack([d, np.zeros((len(d), 1))])
    b[-1] -= alpha*beta/b[0]
    ud[[0, -1], -1] = -b[0], alpha

    #solve the cyclic tri system
    trislv(abc, ud)
    d[:] = ud[:, :-1]
    u = ud[:, -1]

    #compute the final solution
    fact = (d[0] - alpha*d[-1]/b[0])/(1.0 + u[0] - alpha*u[-1]/b[0])
    d -= fact*u[:, None]


def linintrp(x_intrp, xs, ys):
    y_interp = np.zeros((ys.shape[1], x_intrp.shape[0]))
    for i, yss in enumerate(ys.T):
        y_interp[i] = np.interp(x_intrp, xs, yss)

    return y_interp.T 


class CubicSplineFit:
    def __init__(self, xs=None, ys=None, bctype='not-a-knot'):
        
        self.splineslv =  {
            'periodic': lambda abc, d: self._periodic(abc, d), 
            'not-a-knot': lambda abc, d: self._notaknot(abc, d)
            }

        self._bctype = bctype
        if np.any(xs):
            self._construct(xs, ys, self._bctype)
                
    def update(self, xs=None, ys=None, bctype='not-a-knot'):
        compute = np.any(xs) or (not (bctype == self._bctype))
        self._bctype = bctype   
        if compute:
            if not np.any(xs):
                xs, ys, _ = self.spl
            self._construct(xs, ys, self._bctype)

    def _construct(self, xs, ys,  bctype='not-a-knot'):
        self._xs = xs
        self._ys = ys
        abc, d = self._spl_matrix_coeffs()

        self.splineslv[bctype](abc, d)
        self._zs = d

    def __call__(self, x_intrp, xs=None, ys=None):
        if np.any(xs):
            self.update(xs, ys, self._bctype)
        return self._spleval(x_intrp)

    @property
    def spl(self):
        return (self._xs, self._ys, self._zs)
  
    def _spl_matrix_coeffs(self):
        t, y = self._xs, self._ys
        b = np.zeros_like(t)
        d = np.zeros_like(y)
        dx = np.diff(t)
        if len(y.shape) > 1:
            dxs = dx[:, None]
            slope = np.diff(y, axis=0)/dxs  
        else:
            slope = np.diff(y)/dx

        #preparing the core part of the matrix
        b[1: -1] = 2*(dx[: -1] + dx[1: ])  #diag
        a = dx.copy() #lower diag
        c = dx.copy() #upper diag
        d[1: -1] = 6*(slope[1: ] - slope[ :-1]) #rhs
        d[0] = slope[0]
        d[-1] = slope[-1]

        return [a, b, c], d

    def _notaknot(self, abc, d):
        #d3S0(x[1]) = d3S1(x[1])
        #d3S_n-3(x[-2]) = d3S_n-2(x[-2])
        a, b, c = abc
        beta0  = a[0]/a[1]
        beta1  = a[-1]/a[-2]

        b[1]  += beta0*(a[0] + a[1])
        c[1]  -= a[0]*beta0
        b[-2] += beta1*(a[-1] + a[-2])
        a[-2] -= a[-1]*beta1

        abc_m = [a[1: -1], b[1: -1], c[1: -1]]
        trislv(abc_m, d[1: -1])
        d[0]  = -beta0*d[2]  + (1 + beta0)*d[1] 
        d[-1] = -beta1*d[-3] + (1 + beta1)*d[-2]

    def _periodic(self, abc, d):
        #z0 = z1
        #write equation for i=0 and note that i-1=n-2
        a, b, c = abc
        d[0] = 6*(d[0] - d[-1])
        b[0] = 2*(a[0] + a[-1])
        abc_m = [a[: -1], b[: -1], c[: -1]]
        trislvcyc(abc_m, d[: -1], a[-1], a[-1])
        d[-1] = d[0]

    #Evaluate yy at xx point using the cubic spline representation
    def _spleval(self, xarr):
        t, y, z = self.spl
        idxs = get_interval_idxs(t, xarr) 
        t = t[:, None]
        ht = t[idxs+1] - t[idxs]
        hxt = xarr[:, None] - t[idxs]
        yarr = hxt*(0.5*z[idxs] + hxt*(z[idxs+1] - z[idxs])/(6*ht))
        yarr += -ht*(z[idxs+1] + 2*z[idxs])/6 + (y[idxs+1] - y[idxs])/ht
        yarr = y[idxs] + hxt*yarr
        return yarr