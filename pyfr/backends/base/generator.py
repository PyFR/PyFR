# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
import re
from textwrap import dedent

import numpy as np


def funcsig(mods, rett, name, args):
    # Delimit such that we have one (indented) argument per line
    sep = ',\n{}'.format(' '*(len(name) + 1))
    sig = sep.join(args).split('\n')

    # Add in the name and parentheses
    sig[0] = '{}({}'.format(name, sig[0])
    sig[-1] = '{})'.format(sig[-1])

    # Include modifiers and return type
    return ['{} {}'.format(mods, rett)] + sig


def procbody(body, fpdtype):
    # At single precision suffix all floating point constants by 'f'
    if fpdtype == np.float32:
        return re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                      r'\g<0>f', body)
    else:
        return body


class Arg(object):
    def __init__(self, name, spec, body):
        self.name = name

        specptn = (r'(?:(in|inout|out)\s+)?'       # Intent
                   r'(?:(view|scalar)\s+)?'        # Attrs
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

        # See if we are used
        self.isused = bool(re.search(usedptn, body))

        # Currently scalar arguments must be of type fpdtype_t
        if self.isscalar and self.dtype != 'fpdtype_t':
            raise ValueError('Scalar arguments must be of type fpdtype_t')


    @property
    def isview(self):
        return 'view' in self.attrs

    @property
    def isscalar(self):
        return 'scalar' in self.attrs

    @property
    def isvector(self):
        return not self.isscalar

    @property
    def ncdim(self):
        return len(self.cdims)


class BaseFunctionGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, args, rett, body, fpdtype):
        self.name = name
        self.rett = rett
        self.args = arg
        self.body = procbody(body, fpdtype).split('\n')

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
        self.body = procbody(dedent(body), fpdtype).split('\n')

        # Parse and sort our argument list
        sargs = sorted((k, Arg(k, v, body)) for k, v in args.iteritems())

        # Break arguments into point-scalars and point-vectors
        self.scalargs = [v for k, v in sargs if v.isscalar]
        self.vectargs = [v for k, v in sargs if v.isvector]

        # If we are 2D ensure none of our arguments are views
        if ndim == 2 and any(v.isview for v in self.vectargs):
            raise ValueError('View arguments are not supported for 2D kernels')

    @abstractmethod
    def argspec(self):
        pass

    @abstractmethod
    def render(self):
        pass

