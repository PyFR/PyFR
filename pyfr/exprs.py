import itertools as it
import re

import numpy as np


_npeval_syms = {
    '__builtins__': {},
    'exp': np.exp, 'log': np.log,
    'sin': np.sin, 'asin': np.arcsin,
    'cos': np.cos, 'acos': np.arccos,
    'tan': np.tan, 'atan': np.arctan, 'atan2': np.arctan2,
    'abs': np.abs, 'pow': np.power, 'sqrt': np.sqrt, 'cbrt': np.cbrt,
    'tanh': np.tanh, 'pi': np.pi,
    'max': np.maximum, 'min': np.minimum
}


def expr_vars(expr):
    # Free variable names referenced by an expression;
    return sorted(set(re.findall(r'(?<![\d.])[A-Za-z_]\w*', expr))
                  - set(_npeval_syms))


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


def subst_expr_vars(expr, subs):
    # Substitute whole-word variable names with parenthesised expressions
    if not subs:
        return expr

    p = '|'.join(re.escape(k) for k in subs)
    return re.sub(rf'\b({p})\b', lambda m: f'({subs[m[1]]})', expr)


def resolve_aux(aux):
    # Expand references to earlier definitions so each is self-contained
    res = {}
    for k, v in aux.items():
        res[k] = subst_expr_vars(v, res)

    return res
