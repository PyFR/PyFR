from collections import namedtuple
import re

import numpy as np

from pyfr.exprs import subst_expr_vars
from pyfr.inifile import process_expr
from pyfr.stats.finalise import eval_algebraic
from pyfr.stats.provider import StatsTables


TavgExprs = namedtuple('TavgExprs', ['avgs', 'derived', 'alg', 'hidden',
                                     'spec', 'aliases'])


def _merge_exprs(exprs, new, proc, what):
    # Merge generated expressions with the user-provided ones
    norm = lambda e: ''.join(e.split())
    byexpr = {norm(e): n for n, e in exprs.items()}
    merged, aliases = dict(exprs), {}

    for name, expr in new.items():
        pexpr = proc(expr)
        nexpr = norm(pexpr)

        # Same name; require an identical expression
        if name in merged:
            if norm(merged[name]) != nexpr:
                raise ValueError(f'{what}-{name} conflicts with statistics '
                                 f'package expression for {name}')
        # Same expression under a different name; alias it
        elif nexpr in byexpr:
            aliases[name] = byexpr[nexpr]
        else:
            merged[name] = pexpr
            byexpr[nexpr] = name

    return merged, aliases


def expand_exprs(cfg, ndims, elementscls, spec, avgs, funs):
    c = cfg.items_as('constants', float)
    aux = elementscls.auxvars(ndims, cfg)
    tabs = StatsTables(elementscls.stats_tables(cfg), ndims, 'tavg')
    groups, alg = tabs.expand(cfg, spec)

    # Merge the package fields with the user expressions
    proc = lambda e: process_expr(e, c | aux)
    avgs, aliases = _merge_exprs(avgs, groups['field'], proc, 'avg')

    # Constants for derived quantities; field names take precedence
    # and porder stays symbolic so expressions are order-independent
    cexp = {k: v for k, v in c.items() if k not in groups['field']}
    eproc = lambda e: process_expr(subst_expr_vars(e, aliases), cexp)

    # Alias rewriting must shield cross-references between derived
    # quantities since dashes act as word boundaries for substitution
    keys = sorted({**groups['derived'], **groups['derived-hidden']},
                  key=len, reverse=True)

    def shield(e):
        for i, n in enumerate(keys):
            e = re.sub(rf'\b{re.escape(n)}\b', f'xp{i}xp', e)
        e = eproc(e)
        for i, n in enumerate(keys):
            e = e.replace(f'xp{i}xp', n)
        return e

    # Merge the derived expressions with any user expressions
    derived, _ = _merge_exprs(funs, groups['derived'], shield, 'fun-avg')
    hidden = {k: shield(e) for k, e in groups['derived-hidden'].items()}

    # User expressions are always algebraic in the fields
    alg = alg | set(funs)

    return avgs, derived, hidden, alg, aliases


def eval_algebraic_avgs(te, fields):
    # Derived statistics evaluable from the average fields alone
    alg = {n: e for n, e in te.derived.items() if n in te.alg}
    hidden = {n: e for n, e in te.hidden.items() if n in te.alg}

    return eval_algebraic(alg, fields, hidden)


def tavg_exprs(cfg, cfgsect, ndims, elementscls):
    c = cfg.items_as('constants', float)
    aux = elementscls.auxvars(ndims, cfg)

    # Field expressions followed by user functional expressions
    avgs = {k.removeprefix('avg-'): cfg.getexpr(cfgsect, k, subs=c | aux)
            for k in cfg.items(cfgsect, prefix='avg-')}
    funs = {k.removeprefix('fun-avg-'): cfg.getexpr(cfgsect, k, subs=c)
            for k in cfg.items(cfgsect, prefix='fun-avg-')}

    # Expand any requested statistics packages
    spec = cfg.get(cfgsect, 'stats') if cfg.hasopt(cfgsect, 'stats') else None
    derived, hidden, aliases = dict(funs), {}, {}
    alg = set(funs)
    if spec:
        avgs, derived, hidden, alg, aliases = expand_exprs(
            cfg, ndims, elementscls, spec, avgs, funs
        )

    return TavgExprs(avgs, derived, alg, hidden, spec, aliases)
