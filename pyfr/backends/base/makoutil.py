# -*- coding: utf-8 -*-

from mako.runtime import supports_caller, capture

import pyfr.util as util


def ndrange(context, *args):
    return util.ndrange(*args)


def npdtype_to_ctype(context, dtype):
    return nputil.npdtype_to_ctype(dtype)


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
def function(context, name, args, rett='void'):
    # Capture the function body
    body = capture(context, context['caller'].body)

    # Get the generator class and floating point data type
    funcgen, fpdtype = context['_function_generator'], context['fpdtype']

    # Render the complete function
    return funcgen(name, args, rett, body, fpdtype).render()


def alias(context, name, newname):
    return '#define {} {}'.format(newname, name)
