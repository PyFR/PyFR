import re

import numpy as np

from pyfr.exprs import expr_vars, npeval


def _shield_prior(expr, out, subs):
    # Rewrite references to prior outputs, longest names first
    for i, n in enumerate(sorted(out, key=len, reverse=True)):
        if re.search(rf'\b{re.escape(n)}\b', expr):
            expr = re.sub(rf'\b{re.escape(n)}\b', f'oref{i}', expr)
            subs[f'oref{i}'] = out[n]

    return expr


def _eval_in(expr, ns):
    # Substitute known names, longest first, to shield any dashes
    subs = {}
    expr = _shield_prior(expr, ns, subs)

    # Remaining symbols must be simple names in the namespace
    for tok in expr_vars(expr):
        if tok not in subs:
            subs[tok] = ns[tok]

    # Ratios are NaN where the statistics vanish
    with np.errstate(divide='ignore', invalid='ignore'):
        return npeval(expr, subs)


def eval_algebraic(derived, fields, hidden=None):
    ns, out = dict(fields), {}

    # Hidden quantities join the evaluation namespace
    for name, expr in (hidden or {}).items():
        ns[name] = _eval_in(expr, ns)

    for name, expr in derived.items():
        ns[name] = out[name] = _eval_in(expr, ns)

    return out


def eval_exports(exports, data, hidden=None):
    out, cache = {}, {}
    hidden = hidden or {}

    # Resolve a symbol to the geometry, hidden quantity or field it names
    def resolve(tok):
        if m := re.fullmatch(r'grad_(\w+?)_([xyz])', tok):
            return data.grad_avg(m[1])['xyz'.index(m[2])]
        elif m := re.fullmatch(r'lap_(\w+)', tok):
            return data.lap_avg(m[1])
        elif m := re.fullmatch(r'grid_h(min|max)', tok):
            return data.grid_h(m[1] == 'max')
        elif tok == 'boundary_dist':
            return data.boundary_dist
        elif tok in hidden:
            dsub = {t: data.avg(t) for t in expr_vars(hidden[tok])}
            return npeval(hidden[tok], dsub)
        else:
            return data.avg(tok)

    # Symbols which are always available
    base = {'porder': data.porder}
    if (norms := getattr(data, 'normals', None)) is not None:
        base |= {f'norm_{x}': n for x, n in zip('xyz', norms)}

    for name, expr in exports.items():
        subs = dict(base)
        expr = _shield_prior(expr, out, subs)

        # Resolve the remaining symbols, caching across expressions
        for tok in expr_vars(expr):
            if tok not in subs:
                if tok not in cache:
                    cache[tok] = resolve(tok)
                subs[tok] = cache[tok]

        # Ratios are NaN where the statistics vanish
        with np.errstate(divide='ignore', invalid='ignore'):
            out[name] = npeval(expr, subs)

    return out
