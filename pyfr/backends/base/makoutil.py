from collections.abc import Iterable
import itertools as it
import re

from mako.runtime import supports_caller, capture
import numpy as np

import pyfr.nputil as nputil
import pyfr.util as util


def ndrange(context, *args):
    return util.ndrange(*args)


def ilog2range(context, x):
    return [2**i for i in range(x.bit_length() - 2, -1, -1)]


def npdtype_to_ctype(context, dtype):
    return nputil.npdtype_to_ctype(dtype)


def dot(context, a_, b_=None, /, **kwargs):
    ix, nd = util.first(kwargs.items())
    ab = '({})*({})'.format(a_, b_ or a_)

    # Allow for flexible range arguments
    nd = nd if isinstance(nd, Iterable) else [nd]

    return '(' + ' + '.join(ab.format(**{ix: i}) for i in range(*nd)) + ')'


def array(context, expr_, vals_={}, /, **kwargs):
    ix = util.first(kwargs)
    ni = kwargs.pop(ix)
    items = []

    # Allow for flexible range arguments
    for i in range(*(ni if isinstance(ni, Iterable) else [ni])):
        if kwargs:
            items.append(array(context, expr_, vals_ | {ix: i}, **kwargs))
        else:
            items.append(expr_.format_map(vals_ | {ix: i}))

    return '{ ' + ', '.join(items) + ' }'


def polyfit(context, f, a, b, n, var, nqpts=500):
    x = np.linspace(a, b, nqpts)
    y = f(x)

    coeffs = np.polynomial.polynomial.polyfit(x, y, n)
    pfexpr = f' + {var}*('.join(str(c) for c in coeffs) + ')'*n

    return f'({pfexpr})'


def _strip_parens(s):
    out, depth = [], 0

    for c in s:
        depth += (c in '{(') - (c in ')}')

        if depth == 0 and c not in ')}':
            out.append(c)

    return ''.join(out)


def _locals(body):
    # First, strip away any comments
    body = re.sub(r'//.*?\n', '', body)

    # Next, find all variable declaration statements
    decls = re.findall(r'(?:[A-Za-z_]\w*)\s+([A-Za-z_]\w*[^;]*?);', body)

    # Strip anything inside () or {}
    decls = [_strip_parens(d) for d in decls]

    # A statement can define multiple variables, so split by ','
    decls = it.chain.from_iterable(d.split(',') for d in decls)

    # Extract the variable names
    lvars = [re.match(r'\s*(\w+)', v)[1] for v in decls]

    # Prune invalid names
    return [lv for lv in lvars if lv != 'if']


@supports_caller
def macro(context, name, params, externs=''):
    # Check we have not already been defined
    if name in context['_macros']:
        raise RuntimeError(f'Attempt to redefine macro "{name}"')

    # Split up the parameter and external variable list
    params = [p.strip() for p in params.split(',')]
    externs = [e.strip() for e in externs.split(',')] if externs else []

    # Ensure no invalid characters in params/extern variables
    for p in it.chain(params, externs):
        if not re.match(r'[A-Za-z_]\w*$', p):
            raise ValueError(f'Invalid param "{p}" in macro "{name}"')

    # Capture the function body
    body = capture(context, context['caller'].body)

    # Identify any local variable declarations
    lvars = _locals(body)

    # Suffix these variables by a '_'
    if lvars:
        body = re.sub(r'\b({0})\b'.format('|'.join(lvars)), r'\1_', body)

    # Save
    context['_macros'][name] = (params, externs, body)

    return ''


def expand(context, name, /, *args, **kwargs):
    # Get the macro parameter list and the body
    mparams, mexterns, body = context['_macros'][name]

    # Ensure an appropriate number of arguments have been passed
    if len(mparams) != len(args) + len(kwargs):
        raise ValueError(f'Inconsistent macro parameter list in {name}')

    # Parse the parameter list
    params = dict(zip(mparams, args))
    for k, v in kwargs.items():
        if k in params:
            raise ValueError(f'Duplicate macro parameter {k} in {name}')

        params[k] = v

    # Ensure all parameters have been passed
    if sorted(mparams) != sorted(params):
        raise ValueError(f'Inconsistent macro parameter list in {name}')

    # Ensure all (used) external parameters have been passed to the kernel
    for extrn in mexterns:
        if (extrn not in context['_extrns'] and
            re.search(rf'\b{extrn}\b', body)):
            raise ValueError(f'Missing external {extrn} in {name}')

    # Rename local parameters
    for lname, subst in params.items():
        body = re.sub(rf'\b{lname}\b', str(subst), body)

    return f'{{\n{body}\n}}'


@supports_caller
def kernel(context, name, ndim, **kwargs):
    extrns = context['_extrns']

    # Validate the argument list
    if any(arg in extrns for arg in kwargs):
        raise ValueError('Duplicate argument in {0}: {1} {2}'
                         .format(name, list(kwargs), list(extrns)))

    # Merge local and external arguments
    kwargs = dict(kwargs, **extrns)

    # Capture the kernel body
    body = capture(context, context['caller'].body)

    # Get the generator class and data types
    kerngen = context['_kernel_generator']
    fpdtype, ixdtype = context['fpdtype'], context['ixdtype']

    # Instantiate
    kern = kerngen(name, int(ndim), kwargs, body, fpdtype, ixdtype)

    # Save the argument/type list for later use
    context['_kernel_argspecs'][name] = kern.argspec()

    # Render and return the complete kernel
    return kern.render()


def alias(context, name, func):
    context['_macros'][name] = context['_macros'][func]
    return ''
