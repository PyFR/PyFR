from collections import namedtuple
from collections.abc import Iterable
import inspect
import itertools as it
import re

from mako.runtime import Undefined, capture, supports_caller
from mako.template import Template
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


Macro = namedtuple('Macro', ['params', 'externs', 'pyparams', 'caller'])


@supports_caller
def macro(context, name, params, externs=''):
    # Check if already registered - if so, skip
    if name in context['_macros']:
        return ''

    # Parse params to separate params from py:params
    pyparams = [p.strip().removeprefix('py:') for p in params.split(',')
                if p.strip().startswith('py:')]
    params = [p.strip() for p in params.split(',')
              if not p.strip().startswith('py:')]
    externs = [e.strip() for e in externs.split(',')] if externs else []

    # Validate all user created parameter names
    for p in it.chain(params, externs):
        if not re.match(r'[A-Za-z_]\w*$', p):
            raise ValueError(f'Invalid param "{p}" in macro "{name}"')

    # Check if callable has arguments
    sig = inspect.signature(context['caller'].body)
    sigargs = list(sig.parameters.keys())

    # Determine what to do based on pyparams and callable signature
    if pyparams and not sigargs:
        # First pass: pyparams exist but callable has no args yet
        # Need to recreate macro with args='...'
        template = context._with_template

        # Debug: Verify what we have access to
        print(f"\nDEBUG: Finding macro '{name}'", flush=True)
        print(f"DEBUG: template.uri = {template.uri}", flush=True)
        print(f"DEBUG: First 300 chars of template.source:\n{template.source[:300]}", flush=True)

        # Extract the raw macro body text
        rawbody = context.lookup.get_raw_macro(template.source, name)

        # Extract namespace directives (but NOT includes, to avoid re-registering macros)
        namespaces = re.findall(r'<%namespace[^>]+/>', template.source)
        header = '\n'.join(namespaces)

        # Recreate the macro with args='...'
        psrt = ','.join(params)
        esrt = ','.join(externs)
        astr = ', '.join(pyparams)
        mstring = f'''{header}

<%pyfr:macro name='{name}' params='{psrt}' externs='{esrt}' args='{astr}'>
{rawbody}
</%pyfr:macro>'''

        # Render the new template - this will call macro()
        # again with sigargs populated
        new_template = Template(mstring, lookup=context.lookup)
        new_template.render_context(context)

        # After rendering, the macro should be registered by the above render
        if name not in context['_macros']:
            raise ValueError(f'Macro "{name}" failed to re-render')

        return ''

    # If we get here, either:
    # - Second pass (pyparams + sigargs): use signature args
    # - No pyparams used: just save normally
    if sigargs:
        pyparams = sigargs

    context['_macros'][name] = Macro(params, externs, pyparams,
                                     context['caller'].body)
    return ''


def expand(context, name, /, *args, **kwargs):
    macrodef = context['_macros'][name]
    mparams = macrodef.params
    mexterns = macrodef.externs
    mpyparams = macrodef.pyparams
    mcaller = macrodef.caller

    # Separate kwargs into C params and Python data params
    paramskw = {}
    pyparamskw = {}

    for k, v in kwargs.items():
        if k in mpyparams:
            pyparamskw[k] = v
        elif k in mparams:
            paramskw[k] = v
        else:
            raise ValueError(f'Unknown parameter "{k}" in macro "{name}"')

    # Build params dict from positional args and kwargs
    # Positional args fill in params first, then any remaining go to pyparams
    nparamspos = len(mparams) - len(paramskw)
    params = dict(zip(mparams, args[:nparamspos]))
    params.update(paramskw)

    # Check we got all params
    if len(params) != len(mparams):
        raise ValueError(f'Incomplete or duplicate parameters in "{name}"')

    # Build pyparams dict: remaining positional args + kwargs
    # Positional args fill in mpyparams in order
    pyparamspos = args[nparamspos:]
    pyparams = {}

    # Fill in positional pyparams first (in order from mpyparams)
    posidx = 0
    for pyparam in mpyparams:
        if pyparam in pyparamskw:
            # Provided as keyword arg
            pyparams[pyparam] = pyparamskw[pyparam]
        elif posidx < len(pyparamspos):
            # Provided as positional arg
            pyparams[pyparam] = pyparamspos[posidx]
            posidx += 1

    # Check we got all pyparams
    if len(pyparams) != len(mpyparams):
        raise ValueError(f'Incomplete or duplicate Python data parameters '
                         f'in "{name}"')

    # Check all python data is defined
    for k, v in pyparams.items():
        if isinstance(v, Undefined):
            raise ValueError(f'Undefined Python data parameter "{k}" '
                             f'passed to "{name}"')

    # Call macro callable with Python data
    body = capture(context, mcaller, **pyparams)

    # Identify any local variable declarations
    lvars = _locals(body)

    # Suffix these variables by a '_'
    if lvars:
        body = re.sub(r'\b({0})\b'.format('|'.join(lvars)), r'\1_', body)

    # Ensure all (used) external parameters have been passed to the kernel
    for extrn in mexterns:
        if (extrn not in context['_extrns'] and
            re.search(rf'\b{extrn}\b', body)):
            raise ValueError(f'Missing external "{extrn}" in "{name}"')

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
