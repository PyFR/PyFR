# -*- coding: utf-8 -*-

from collections import Iterable
import itertools as it
import re

from mako.runtime import supports_caller, capture

import pyfr.nputil as nputil
import pyfr.util as util


def ndrange(context, *args):
    return util.ndrange(*args)


def npdtype_to_ctype(context, dtype):
    return nputil.npdtype_to_ctype(dtype)


def dot(context, a_, b_=None, **kwargs):
    ix, nd = next(kwargs.iteritems())
    ab = '({})*({})'.format(a_, b_ or a_)

    # Allow for flexible range arguments
    nd = nd if isinstance(nd, Iterable) else [nd]

    return '(' + ' + '.join(ab.format(**{ix: i}) for i in xrange(*nd)) + ')'


def array(context, ex_, **kwargs):
    ix, ni = next(kwargs.iteritems())

    # Allow for flexible range arguments
    ni = ni if isinstance(ni, Iterable) else [ni]

    return '{ ' + ', '.join(ex_.format(**{ix: i}) for i in xrange(*ni)) + ' }'


def _locals(body):
    # First, strip away any comments
    body = re.sub(r'//.*?\n', '', body)

    # Next, find all variable declaration statements
    decls = re.findall(r'(?:[A-Za-z_]\w*)\s+([A-Za-z_]\w*[^;]*?);', body)

    # A statement can define multiple variables, so split by ','
    decls = (re.split(r'\s*,\s*(?![^{}]+\})', d) for d in decls)
    decls = it.chain.from_iterable(decls)

    return [re.match(r'\w+', v).group(0) for v in decls]


@supports_caller
def macro(context, name, params):
    # Check we have not already been defined
    if name in context['_macros']:
        raise RuntimeError('Attempt to redefine macro "{0}"'
                           .format(name))

    # Split up the parameter list
    params = [p.strip() for p in params.split(',')]

    # Capture the function body
    body = capture(context, context['caller'].body)

    # Identify any local variable declarations
    lvars = _locals(body)

    # Suffix these variables by a '_'
    if lvars:
        body = re.sub(r'\b({0})\b'.format('|'.join(lvars)), r'\1_', body)

    # Save
    context['_macros'][name] = (params, body)

    return ''


def expand(context, name, *params):
    # Get the macro parameter list and the body
    mparams, body = context['_macros'][name]

    # Validate
    if len(mparams) != len(params):
        raise ValueError('Invalid macro parameter list')

    # Substitute
    for name, subst in zip(mparams, params):
        body = re.sub(r'\b{0}\b'.format(name), subst, body)

    return '{\n' + body + '\n}'


@supports_caller
def kernel(context, name, ndim, **kwargs):
    # Capture the kernel body
    body = capture(context, context['caller'].body)

    # Get the generator class and floating point data type
    kerngen, fpdtype = context['_kernel_generator'], context['fpdtype']

    # Instantiate
    kern = kerngen(name, int(ndim), kwargs, body, fpdtype)

    # Save the argument/type list for later use
    context['_kernel_argspecs'][name] = kern.argspec()

    # Render and return the complete kernel
    return kern.render()


def alias(context, name, func):
    context['_macros'][name] = context['_macros'][func]
    return ''
