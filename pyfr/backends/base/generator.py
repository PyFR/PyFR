# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
import re
from textwrap import dedent

import numpy as np


def funccall(name, args):
    # Delimit such that we have one (indented) argument per line
    sep = ',\n' + ' '*(len(name) + 1)
    cal = sep.join(args).split('\n')

    # Add in the name and parentheses
    cal[0] = '{}({}'.format(name, cal[0])
    cal[-1] = '{})'.format(cal[-1])

    return cal


def funcsig(mods, rett, name, args):
    # Exploit the similarity between prototypes and invocations
    sig = funccall(name, args)

    # Include modifiers and return type
    return ['{} {}'.format(mods, rett).strip()] + sig


def procbody(body, fpdtype):
    # Remove any indentation
    body = dedent(body)

    # At single precision suffix all floating point constants by 'f'
    if fpdtype == np.float32:
        body = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                      r'\g<0>f', body)

    # Split into lines
    return body.split('\n')


class Arg(object):
    def __init__(self, name, spec, body):
        self.name = name

        specptn = (r'(?:(in|inout|out)\s+)?'       # Intent
                   r'(?:(mpi|scalar|view)\s+)?'    # Attrs
                   r'([A-Za-z_]\w*)'               # Data type
                   r'((?:\[\d+\]){0,2})$')         # Constant array dimensions
        dimsptn = r'(?<=\[)\d+(?=\])'
        usedptn = r'(?:[^A-Za-z]|^){}[^A-Za-z0-9]'.format(name)

        # Parse our specification
        m = re.match(specptn, spec)
        if not m:
            raise ValueError('Invalid argument specification')

        g = m.groups()

        # Properties
        self.intent = g[0] or 'in'
        self.attrs = g[1] or ''
        self.dtype = g[2]

        # Dimensions
        self.cdims = [int(d) for d in re.findall(dimsptn, g[3])]
        self.ncdim = len(self.cdims)

        # Attributes
        self.ismpi = 'mpi' in self.attrs
        self.isused = bool(re.search(usedptn, body))
        self.isview = 'view' in self.attrs
        self.isscalar = 'scalar' in self.attrs
        self.isvector = 'scalar' not in self.attrs

        # Currently scalar arguments must be of type fpdtype_t
        if self.isscalar and self.dtype != 'fpdtype_t':
            raise ValueError('Scalar arguments must be of type fpdtype_t')


class BaseFunctionGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, params, rett, body, fpdtype):
        self.name = name
        self.rett = rett
        self.args = re.split(r'\s*,\s*', params)
        self.body = procbody(body, fpdtype)[1:-1]

    def render(self):
        head = funcsig(self.mods, self.rett, self.name, self.args)
        body = [' '*4 + l for l in self.body]

        # Add in the '{' and '}' and concat
        return '\n'.join(head + ['{'] + body + ['}'])

    @abstractproperty
    def mods(self):
        pass


class BaseKernelGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, ndim, args, body, fpdtype):
        self.name = name
        self.ndim = ndim
        self.body = procbody(body, fpdtype)
        self.fpdtype = fpdtype

        # Parse and sort our argument list
        sargs = sorted((k, Arg(k, v, body)) for k, v in args.iteritems())

        # Eliminate unused arguments
        sargs = [v for k, v in sargs if v.isused]

        # Break arguments into point-scalars and point-vectors
        self.scalargs = [v for v in sargs if v.isscalar]
        self.vectargs = [v for v in sargs if v.isvector]

        # If we are 2D ensure none of our arguments are views
        if ndim == 2 and any(v.isview for v in self.vectargs):
            raise ValueError('View arguments are not supported for 2D kernels')

        # Similarly, check for MPI matrices
        if ndim == 2 and any(v.ismpi for v in self.vectargs):
            raise ValueError('MPI matrices are not supported for 2D kernels')

    @abstractmethod
    def argspec(self):
        pass

    @abstractmethod
    def render(self):
        pass
