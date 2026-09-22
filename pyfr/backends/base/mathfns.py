from math import factorial, gcd

import numpy as np
from mako.template import Template
from numpy.polynomial.legendre import leg2poly, legfit, leggauss
from numpy.polynomial.polynomial import polyfit

from pyfr.dsl.nodes import (Binary, Call, Int, Number, Region, Unary, Var,
                            map_ast)
from pyfr.dsl.simplifier import expand_power, expr_size, reduce_pow


def _galerkin_coeffs(q, deg=4, nq=20):
    # L2 Galerkin projection of m^(-1/q) onto Legendre basis on [1,2)
    nodes, wts = leggauss(nq)
    f = (0.5*nodes + 1.5)**(-1.0/q)

    # Fit in Legendre basis then convert to monomials in u = m - 1.5
    lc = legfit(nodes, f, deg, w=np.sqrt(wts))
    mc = leg2poly(lc)
    return (mc * 2.0**np.arange(len(mc))).tolist()


def _scale_coeffs(q):
    # Lagrange interpolation of 2^(-r/q) for r = 0..q-1
    r = np.arange(q, dtype=np.float64)
    return polyfit(r, 2.0**(-r/q), q - 1).tolist()


def _horner(coeffs, var, sfx):
    s = f'{coeffs[-1]:.18e}{sfx}'
    for c in reversed(coeffs[:-1]):
        s = f'{c:.18e}{sfx} + {var}*({s})'
    return s


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

_pow_tpl = Template('''\
${qual}static inline fpdtype_t _pyfr_pow_${r}_${q}(fpdtype_t x)
{
    ${tr}
    union { fpdtype_t f; fp_uint_t u; } _b;
    _b.f = t;
    int _e = (int)((_b.u >> ${eshift}) & ${emask}) - ${ebias};
    _b.u = (_b.u & ${mmask}) | ${one_bits};
    fpdtype_t m = _b.f, u = m - 1.5${sfx};
    fpdtype_t y = ${seed};
    fpdtype_t ef = (fpdtype_t)_e;
    fpdtype_t qf = floor(ef * ${inv_q});
    fpdtype_t rf = ef - ${q}.0${sfx}*qf;
    y *= ${scale};
    _b.u = (fp_uint_t)(${ebias} - (int)qf) << ${eshift};
    y *= _b.f;
% for _ in range(nn):
    { ${yq} y = y*(${q + 1}.0${sfx} - t*yq) * ${inv_q}; }
% endfor
    return 1.0${sfx} / y;
}''')


_exp_tpl = Template('''\
${qual}static inline fpdtype_t _pyfr_exp(fpdtype_t x)
{
    fpdtype_t n = rint${sfx}(x*${log2e});
    fpdtype_t r = fma${sfx}(-n, ${ln2hi}, x);
    r = fma${sfx}(-n, ${ln2lo}, r);
    fpdtype_t p = ${poly};
    union { fpdtype_t f; fp_uint_t u; } _b;
    _b.u = (fp_uint_t)((int)n + ${ebias}) << ${eshift};
    return p*_b.f;
}''')


def generate_exp_fn(fpdtype, qual=''):
    # Split of ln2 into a leading part and a residual correction
    if fpdtype == np.float64:
        sfx = ''
        eshift, ebias = 52, 1023
        ln2hi, ln2lo = 6.931471805599452862e-01, 2.319046813846299558e-17
        ncoef = 14
    else:
        sfx = 'f'
        eshift, ebias = 23, 127
        ln2hi, ln2lo = 6.9314575195e-01, 1.4286067653e-06
        ncoef = 8

    # Taylor series for exp(r) on [-ln2/2, ln2/2]
    ecoef = [1/factorial(k) for k in range(ncoef)]

    return _exp_tpl.render(
        qual=qual, sfx=sfx, eshift=eshift, ebias=ebias,
        log2e=f'{1.4426950408889634:.18e}{sfx}',
        ln2hi=f'{ln2hi:.18e}{sfx}', ln2lo=f'{ln2lo:.18e}{sfx}',
        poly=_horner(ecoef, 'r', sfx),
    ).rstrip()


def generate_pow_fn(r, q, fpdtype, qual=''):
    g = gcd(r, q)
    r, q = r // g, q // g

    if fpdtype == np.float64:
        sfx = ''
        eshift, emask = 52, '0x7FF'
        ebias = 1023
        mmask = '0x000FFFFFFFFFFFFFULL'
        one_bits = '0x3FF0000000000000ULL'
        nn = 2
    else:
        sfx = 'f'
        eshift, emask = 23, '0xFF'
        ebias = 127
        mmask = '0x007FFFFF'
        one_bits = '0x3F800000'
        nn = 1

    mc = _galerkin_coeffs(q)
    sc = _scale_coeffs(q)

    return _pow_tpl.render(
        qual=qual, r=r, q=q, sfx=sfx, nn=nn,
        tr=_TR[r], yq=_YQ[q],
        eshift=eshift, emask=emask, ebias=ebias,
        mmask=mmask, one_bits=one_bits,
        seed=_horner(mc, 'u', sfx),
        scale=_horner(sc, 'rf', sfx),
        inv_q=f'{1.0/q:.18e}{sfx}',
    ).rstrip()


def _lower_pow(base, v, helpers):
    # Integer and half-integer exponents need only multiplies and a sqrt
    if exact := reduce_pow(base, v):
        return exact

    neg, av = v < 0, abs(v)

    for q in (3, 5, 7):
        p = round(av*q)
        if abs(av*q - p) > 1e-12 or p == 0:
            continue

        k, r = divmod(p, q)
        if r == 0:
            continue

        frac = Call(Var(f'_pyfr_pow_{r}_{q}'), [base])
        if k == 0:
            result = frac
        elif k == 1:
            result = Binary('*', base, frac)
        elif k <= 8 and expr_size(base) <= 3:
            result = Binary('*', expand_power(base, k), frac)
        else:
            continue

        if neg:
            result = Binary('/', Int(1), result)

        helpers.add(('pow', r, q))
        return result

    return None


def generate_helper(key, fpdtype, qual=''):
    # Emit the helper function identified by the given key
    match key:
        case ('exp',):
            return generate_exp_fn(fpdtype, qual)
        case ('pow', r, q):
            return generate_pow_fn(r, q, fpdtype, qual)


def _const_value(expr):
    # Literal exponents may carry a leading unary minus
    match expr:
        case Number(v):
            return v
        case Unary('-', Number(v)):
            return -v
        case _:
            return None


def lower_math_fns(ast, fns, frozen=frozenset({'fp-precise'})):
    helpers = set()

    def lower(node):
        # Leave regions which forbid rearrangement untouched
        if isinstance(node, Region) and node.kind in frozen:
            return node

        node = map_ast(node, lower)
        match node:
            case Call(Var('exp'), [arg]) if 'exp' in fns:
                helpers.add(('exp',))
                return Call(Var('_pyfr_exp'), [arg])
            case Binary('**', base, e):
                if (v := _const_value(e)) is not None:
                    return _lower_pow(base, v, helpers) or node
                else:
                    return node
            case _:
                return node

    # Hand back the lowered AST plus the set of helpers it now requires
    return lower(ast), helpers
