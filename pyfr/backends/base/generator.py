# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
import re

import numpy as np


def procbody(body, fpdtype):
    # At single precision suffix all floating point constants by 'f'
    if fpdtype == np.float32:
        body = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                      r'\g<0>f', body)

    return body


class Arg(object):
    def __init__(self, name, spec, body):
        self.name = name

        specptn = (r'(?:(in|inout|out)\s+)?'       # Intent
                   r'(?:(mpi|scalar|view)\s+)?'    # Attrs
                   r'([A-Za-z_]\w*)'               # Data type
                   r'((?:\[\d+\]){0,2})$')         # Constant array dimensions
        dimsptn = r'(?<=\[)\d+(?=\])'
        usedptn = r'(?:[^A-Za-z]|^){0}[^A-Za-z0-9]'.format(name)

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


class BaseKernelGenerator(object, metaclass=ABCMeta):
    def __init__(self, name, ndim, args, body, fpdtype):
        self.name = name
        self.ndim = ndim
        self.body = procbody(body, fpdtype)
        self.fpdtype = fpdtype

        # Parse and sort our argument list
        sargs = sorted((k, Arg(k, v, body)) for k, v in args.items())

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

    def argspec(self):
        # Argument names and types
        argn, argt = [], []

        # Dimensions
        argn += self._dims
        argt += [[np.int32]]*self.ndim

        # Scalar args (always of type fpdtype)
        argn += [sa.name for sa in self.scalargs]
        argt += [[self.fpdtype]]*len(self.scalargs)

        # Vector args
        for va in self.vectargs:
            argn.append(va.name)

            # View
            if va.isview:
                argt.append([np.intp]*(2 + va.ncdim))
            # Non-stacked vector or MPI type
            elif self.ndim == 1 and (va.ncdim == 0 or va.ismpi):
                argt.append([np.intp])
            # Stacked vector/matrix/stacked matrix
            else:
                argt.append([np.intp, np.int32])

        # Return
        return self.ndim, argn, argt

    @abstractmethod
    def render(self):
        pass
