# -*- coding: utf-8 -*-

from collections import Iterable

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


@supports_caller
def function(context, name, params, rett='void'):
    # Capture the function body
    body = capture(context, context['caller'].body)

    # Get the generator class and floating point data type
    funcgen, fpdtype = context['_function_generator'], context['fpdtype']

    # Render the complete function
    return funcgen(name, params, rett, body, fpdtype).render()


def alias(context, name, func):
    return '#define {} {}'.format(name, func)
