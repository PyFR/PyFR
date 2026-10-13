from decimal import Context, Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.polynomial import polyfit, polyval

from pyfr.dsl.nodes import (PRECISE_REGIONS, Binary, Call, Int, Number,
                            Region, Unary, Var, map_ast)
from pyfr.dsl.simplifier import expand_power, expr_size, reduce_pow
from pyfr.template import DottedTemplateLookup
from pyfr.util import subclasses


def _fsuffix(fpdtype):
    # Suffix for floating point literals of the given type
    return '' if fpdtype == np.float64 else 'f'


def _fp_params(fpdtype):
    # Bit layout of the floating point type and its literal suffixes
    fi = np.finfo(fpdtype)
    eshift, ebias = fi.nmant, fi.maxexp - 1

    return {
        'fsuffix': _fsuffix(fpdtype), 'eshift': eshift, 'ebias': ebias,
        'emask': 2*fi.maxexp - 1, 'mmask': (1 << eshift) - 1,
        'one_bits': ebias << eshift
    }


# Addition chains for y^q (computing yq from y)
_YQ = {
    3: 'fpdtype_t y2 = y*y, yq = y2*y;',
    5: 'fpdtype_t y2 = y*y, y4 = y2*y2, yq = y4*y;',
    7: 'fpdtype_t y2 = y*y, y4 = y2*y2, yq = y4*y2*y;',
}

# Addition chains for t = x^r
_TR = {
    1: 'fpdtype_t t = x;',
    2: 'fpdtype_t t = x*x;',
    3: 'fpdtype_t t = x*x*x;',
    4: 'fpdtype_t x2 = x*x; fpdtype_t t = x2*x2;',
    5: 'fpdtype_t x2 = x*x; fpdtype_t t = x2*x2*x;',
    6: 'fpdtype_t x2 = x*x, x3 = x2*x; fpdtype_t t = x3*x3;',
}


def _const_value(expr):
    # Literal exponents may carry a leading unary minus
    match expr:
        case Number(v):
            return v
        case Unary('-', Number(v)):
            return -v
        case _:
            return None


class BaseMathFn:
    # DSL function this class lowers
    name = None

    # Single argument functions we replace with calls to our helper
    fns = ()

    # Lookup for the templates of our helper functions
    _lookup = DottedTemplateLookup('pyfr.backends.base.kernels.mathfns', {})

    def __init__(self, fn):
        # Name of the generated helper function
        self.cname = f'_pyfr_{fn}'

    @classmethod
    def lower(cls, node, helpers):
        match node:
            case Call(Var(fn), [arg]) if fn in cls.fns:
                return cls(fn).call(helpers, arg)
            case _:
                return None

    def call(self, helpers, *args):
        # Register ourself as a helper and call it with the given arguments
        helpers[self.cname] = self
        return Call(Var(self.cname), list(args))

    def tplargs(self, fpdtype):
        return {}

    def render(self, fpdtype, qual=''):
        fp = _fp_params(fpdtype)
        src = self._lookup.get_template(self.name).render(
            **fp, **self.tplargs(fpdtype), cname=self.cname, qual=qual
        )

        return src.rstrip()

    @staticmethod
    def _horner(coeffs, var, fpdtype):
        fsuffix = _fsuffix(fpdtype)
        s = f'{coeffs[-1]}{fsuffix}'
        for c in reversed(coeffs[:-1]):
            s = f'{c}{fsuffix} + {var}*({s})'
        return s

    @staticmethod
    def _split(v, nbits):
        # Leading nbits of v plus the residual correction
        hi = round(v*2**nbits) / 2**nbits
        return hi, float(v - Decimal(hi))


class ExpMathFn(BaseMathFn):
    name = 'exp'
    fns = ('exp',)

    def tplargs(self, fpdtype):
        fi = np.finfo(fpdtype)

        # Split of ln(2) into a leading part and a residual correction
        with localcontext(Context(prec=40)):
            ln2hi, ln2lo = self._split(Decimal(2).ln(), fi.nmant + 1)

        # Add Taylor terms until the first omitted one is below half an ulp
        rmax, ncoef = math.log(2) / 2, 1
        while rmax**ncoef / math.factorial(ncoef) >= fi.eps / 2:
            ncoef += 1

        # Taylor series for exp(r) on [-ln(2)/2, ln(2)/2]
        ecoef = [1/math.factorial(k) for k in range(ncoef)]

        return {
            'log2e': math.log2(math.e), 'ln2hi': ln2hi, 'ln2lo': ln2lo,
            'poly': self._horner(ecoef, 'r', fpdtype)
        }


class LogMathFn(BaseMathFn):
    name = 'log'

    # Bases of the logarithms we lower with None denoting the natural one
    bases = {'log': None, 'log2': 2, 'log10': 10}
    fns = bases

    def __init__(self, fn):
        super().__init__(fn)
        self.base = self.bases[fn]

    @staticmethod
    def _series_coeffs(eps, nq=20):
        # Fit and check points in t = w/wmax where w = s^2
        wmax = (3 - 2*math.sqrt(2))**2
        nodes, wts = leggauss(nq)
        t, tc = 0.5*(nodes + 1), np.linspace(0, 1, 1001)

        def series(x):
            return sum(2*(wmax*x)**k / (2*k + 3) for k in range(40))

        # Sample the series for (2*atanh(s) - 2*s)/s^3 at these points
        q, qc = series(t), series(tc)

        # Raise the degree until the error in log(m) is below half an ulp
        deg, err = 0, math.inf
        while err >= eps:
            deg += 1
            c = polyfit(t, q, deg, w=t*wts**0.5)
            err = np.max(wmax*tc*np.abs(qc - polyval(tc, c)))

        # Convert the coefficients from powers of t into powers of w
        return (c / wmax**np.arange(deg + 1)).tolist()

    def tplargs(self, fpdtype):
        fi = np.finfo(fpdtype)

        # Fit of the series for 2*atanh(s) - 2*s divided by s^3
        qcoef = self._series_coeffs(fi.eps)

        with localcontext(Context(prec=40)):
            # Natural logarithm of our base
            lnb = Decimal(self.base).ln() if self.base else Decimal(1)

            # Split of log(2) in our base into a leading part and a residual
            lb2hi, lb2lo = self._split(Decimal(2).ln() / lnb, fi.nmant + 1)

            # Factor converting natural logarithms into our base
            lbe = float(1 / lnb)

        return {
            'sqrt2': math.sqrt(2), 'lb2hi': lb2hi, 'lb2lo': lb2lo,
            'lbe': lbe, 'poly': self._horner(qcoef, 'w', fpdtype)
        }


class PowMathFn(BaseMathFn):
    name = 'pow'

    def __init__(self, r, q):
        super().__init__(f'pow_{r}_{q}')
        self.r, self.q = r, q

    @classmethod
    def lower(cls, node, helpers):
        match node:
            case Binary('**', base, e) if (v := _const_value(e)) is not None:
                return cls._lower_pow(base, v, helpers)
            case _:
                return None

    @classmethod
    def _lower_pow(cls, base, v, helpers):
        # Write the magnitude of the exponent as k + r/q
        av = abs(v)
        fv = Fraction(av).limit_denominator(7)
        p, q = fv.numerator, fv.denominator
        k, r = divmod(p, q)

        large = k > 1 and (k > 8 or expr_size(base) > 3)
        badfrac = q not in (3, 5, 7) or abs(av*q - p) > 1e-12

        # Reduce integer and half-integer exponents to multiplies and sqrts
        if exact := reduce_pow(base, v):
            result = exact
        # Pass on large or awkward exponents
        elif large or badfrac:
            result = None
        # Evaluate x^(r/q) with a helper and multiply in x^k
        else:
            result = cls(r, q).call(helpers, base)

            if k == 1:
                result = Binary('*', base, result)
            elif k > 1:
                result = Binary('*', expand_power(base, k), result)

            if v < 0:
                result = Binary('/', Int(1), result)

        return result

    @staticmethod
    def _invroot_coeffs(q, deg=4, nq=20):
        # Sample m^(-1/q) at Gauss-Legendre points mapped to [1, 2]
        nodes, wts = leggauss(nq)
        u = 0.5*nodes

        # Fit a polynomial in u = m - 1.5
        return polyfit(u, (u + 1.5)**(-1 / q), deg, w=wts**0.5).tolist()

    @staticmethod
    def _pow2_coeffs(q):
        # Lagrange interpolation of 2^(-r/q) for r = 0..q-1
        r = np.arange(q, dtype=float)
        return polyfit(r, 2**(-r/q), q - 1).tolist()

    def tplargs(self, fpdtype):
        q = self.q

        # Newton iterations needed to reach full precision
        nn = 2 if fpdtype == np.float64 else 1

        mc = self._invroot_coeffs(q)
        sc = self._pow2_coeffs(q)

        return {
            'q': q, 'nn': nn, 'tr': _TR[self.r], 'yq': _YQ[q],
            'seed': self._horner(mc, 'u', fpdtype),
            'scale': self._horner(sc, 'rf', fpdtype), 'inv_q': 1/q
        }


def lower_math_fns(ast, fns, frozen=PRECISE_REGIONS):
    mfs = {c.name: c for c in subclasses(BaseMathFn, just_leaf=True)}
    mfcls = [mfs[n] for n in fns]
    helpers = {}

    def lower(node):
        # Leave regions which forbid rearrangement untouched
        if isinstance(node, Region) and node.kind in frozen:
            return node

        node = map_ast(node, lower)
        for c in mfcls:
            if (new := c.lower(node, helpers)) is not None:
                return new

        return node

    # Hand back the lowered AST plus the set of helpers it now requires
    return lower(ast), helpers
