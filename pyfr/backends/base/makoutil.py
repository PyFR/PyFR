from collections import namedtuple
from collections.abc import Iterable
from inspect import signature
import itertools as it
import re

from mako.runtime import capture, supports_caller
import numpy as np

import pyfr.nputil as nputil
import pyfr.util as util


class MacroError(Exception):
    def __init__(self, mname, error):
        if isinstance(error, MacroError):
            self.path = [mname, *error.path]
            self.error = error.error
        else:
            self.path = [mname]
            self.error = error

        super().__init__(mname)


class KernelError(Exception):
    def __init__(self, kname, error):
        if isinstance(error, MacroError):
            path = ' -> '.join(it.chain([kname], error.path))
            msg = str(error.error)
        else:
            path = kname
            msg = str(error)

        super().__init__(f'\n  {path}: {msg}')


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


Macro = namedtuple('Macro', ['params', 'externs', 'argsig', 'caller', 'id'])


def mfilttag(source):
    apattern = r'(\w+)=[\'"]([^\'"]*)[\'"]'

    def process_tag(match):
        # Extract all attributes from the opening tag
        attrs = dict(re.findall(apattern, match[1]))

        # Process params if they exist
        if 'params' in attrs:
            params = [p.strip() for p in attrs['params'].split(',')]
            pyparams = [p[3:] for p in params if p.startswith('py:')]
            params = [p for p in params if not p.startswith('py:')]

            attrs['params'] = ', '.join(params)
            if pyparams:
                attrs['args'] = ', '.join(pyparams)

        # Add the ID attribute
        attrs['id'] = util.digest(match[0])

        # Reconstruct the opening tag and append the body
        attrstr = ' '.join(f'{k}="{v}"' for k, v in attrs.items())
        return f'<%pyfr:macro {attrstr}>{match[2]}'

    mpattern = r'(<%pyfr:macro\s+[^>]+>)(.*?</%pyfr:macro>)'
    return re.sub(mpattern, process_tag, source, flags=re.S)


@supports_caller
def macro(context, name, params, externs='', id=''):
    # Check for existing registration and multiple definitions
    if name in context['_macros']:
        # Check for multiple definitions of macro name
        if context['_macros'][name].id != id:
            raise ValueError(f'Attempt to redefine macro "{name}"')
        # Already registered, just return (allow multiple includes)
        return ''

    # Parse and validate params/externs
    params = [p.strip() for p in params.split(',')]
    externs = [e.strip() for e in externs.split(',')] if externs else []

    # Ensure no invalid characters in params/extern variables
    for p in it.chain(params, externs):
        if not re.match(r'[A-Za-z_]\w*$', p):
            raise ValueError(f'Invalid param "{p}" in macro "{name}"')

    # Extract signature from callable for Python variables
    argsig = signature(context['caller'].body)

    # Register the macro with an empty ids set
    context['_macros'][name] = Macro(params, externs, argsig,
                                     context['caller'].body, id)
    return ''


def _parse_expand_args(name, mparams, margsig, args, kwargs):
    margs = list(margsig.parameters.keys())

    # Separate kwargs into params and Python data params
    if unknown := set(kwargs) - set(mparams) - set(margs):
        unknown = unknown.pop()
        raise MacroError(name, f'Unknown parameter "{unknown}"')

    paramskw = {k: v for k, v in kwargs.items() if k in mparams}
    pyparamskw = {k: v for k, v in kwargs.items() if k in margs}

    # Build params dict from positional args and kwargs
    # Positional args fill in params first, then any remaining go to pyparams
    nparamspos = len(mparams) - len(paramskw)
    params = dict(zip(mparams, args[:nparamspos]), **paramskw)

    # Check we got all params
    if len(params) != len(mparams):
        raise MacroError(name, 'Incomplete or duplicate parameters')

    # Parse pyparams
    try:
        bound = margsig.bind(*args[nparamspos:], **pyparamskw)
        pyparams = dict(bound.arguments)
    except TypeError as e:
        raise MacroError(name, f'Invalid Python data parameters: {e}')

    return params, pyparams


def expand(context, name, /, *args, **kwargs):
    mdef = context['_macros'][name]

    # Parse arguments
    params, pyparams = _parse_expand_args(name, mdef.params, mdef.argsig,
                                          args, kwargs)

    # Call macro callable with Python data
    try:
        body = capture(context, mdef.caller, **pyparams)
    except Exception as e:
        raise MacroError(name, e) from e

    # Identify any local variable declarations
    lvars = _locals(body)

    # Suffix these variables by a '_'
    if lvars:
        body = re.sub(r'\b({0})\b'.format('|'.join(lvars)), r'\1_', body)

    # Ensure all (used) external parameters have been passed to the kernel
    for extrn in mdef.externs:
        if (extrn not in context['_extrns'] and
            re.search(rf'\b{extrn}\b', body)):
            raise MacroError(name, f'Missing external "{extrn}"')

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
    try:
        body = capture(context, context['caller'].body)
    except Exception as e:
        raise KernelError(name, e) from e

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
