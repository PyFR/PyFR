# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import re

import numpy as np


class Arg(object):
    def __init__(self, name, spec, body):
        self.name = name

        specptn = (r'(?:(in|inout|out)\s+)?'                # Intent
                   r'(?:(broadcast|mpi|scalar|view)\s+)?'   # Attrs
                   r'([A-Za-z_]\w*)'                        # Data type
                   r'((?:\[\d+\]){0,2})$')                  # Dimensions
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
        self.isbroadcast = 'broadcast' in self.attrs
        self.ismpi = 'mpi' in self.attrs
        self.isused = bool(re.search(usedptn, body))
        self.isview = 'view' in self.attrs
        self.isscalar = 'scalar' in self.attrs
        self.isvector = 'scalar' not in self.attrs

        # Validation
        if self.isbroadcast and self.intent != 'in':
            raise ValueError('Broadcast arguments must be of intent in')
        if self.isbroadcast and self.ncdim != 0:
            raise ValueError('Broadcast arguments can not have dimensions')
        if self.isscalar and self.dtype != 'fpdtype_t':
            raise ValueError('Scalar arguments must be of type fpdtype_t')


class BaseKernelGenerator(object, metaclass=ABCMeta):
    def __init__(self, name, ndim, args, body, fpdtype):
        self.name = name
        self.ndim = ndim
        self.fpdtype = fpdtype

        # Parse and sort our argument list
        sargs = sorted((k, Arg(k, v, body)) for k, v in args.items())

        # Eliminate unused arguments
        sargs = [v for k, v in sargs if v.isused]

        # Break arguments into point-scalars and point-vectors
        self.scalargs = [v for v in sargs if v.isscalar]
        self.vectargs = [v for v in sargs if v.isvector]

        # If we are 1D ensure that none of our arguments are broadcasts
        if ndim == 1 and any(v.isbroadcast for v in self.vectargs):
            raise ValueError('Broadcast arguments are not supported in 1D '
                             'kernels')

        # If we are 2D ensure none of our arguments are views
        if ndim == 2 and any(v.isview for v in self.vectargs):
            raise ValueError('View arguments are not supported for 2D '
                             'kernels')

        # Similarly, check for MPI matrices
        if ndim == 2 and any(v.ismpi for v in self.vectargs):
            raise ValueError('MPI matrices are not supported for 2D kernels')

        # Render the main body of our kernel
        self.body = self._render_body(body)

        # Determine the dimensions to be iterated over
        self._dims = ['_nx'] if ndim == 1 else ['_ny', '_nx']

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
            # Broadcast vector
            elif va.isbroadcast:
                argt.append([np.intp])
            # Stacked vector/matrix/stacked matrix
            else:
                argt.append([np.intp, np.int32])

        # Return
        return self.ndim, argn, argt

    def needs_lsdim(self, arg):
        return ((self.ndim == 2 and not arg.isbroadcast) or
                (arg.ncdim > 0 and not arg.ismpi))

    @abstractmethod
    def render(self):
        pass

    def _deref_arg_view(self, arg):
        ptns = ['{0}_v[{0}_vix[_x]]',
                r'{0}_v[{0}_vix[_x] + {0}_vcstri[_x]*\1]',
                r'{0}_v[{0}_vix[_x] + {0}_vrstri[_x]*\1 + {0}_vcstri[_x]*\2]']

        return ptns[arg.ncdim].format(arg.name)

    def _deref_arg_array_1d(self, arg):
        # Leading (sub) dimension
        lsdim = 'lsd' + arg.name if not arg.ismpi else '_nx'

        # Vector: name_v[_x]
        if arg.ncdim == 0:
            ix = '_x'
        # Stacked vector: name_v[lsdim*\1 + _x]
        elif arg.ncdim == 1:
            ix = r'{0}*\1 + _x'.format(lsdim)
        # Doubly stacked vector: name_v[(nv*\1 + \2)*lsdim + _x]
        else:
            ix = r'({0}*\1 + \2)*{1} + _x'.format(arg.cdims[1], lsdim)

        return '{0}_v[{1}]'.format(arg.name, ix)

    def _deref_arg_array_2d(self, arg):
        # Broadcast vector: name_v[_x]
        if arg.isbroadcast:
            ix = '_x'
        # Matrix: name_v[lsdim*_y + _x]
        elif arg.ncdim == 0:
            ix = 'lsd{}*_y + _x'.format(arg.name)
        # Stacked matrix: name_v[(_y*nv + \1)*lsdim + _x]
        elif arg.ncdim == 1:
            ix = r'(_y*{0} + \1)*lsd{1} + _x'.format(arg.cdims[0], arg.name)
        # Doubly stacked matrix: name_v[((\1*_ny + _y)*nv + \2)*lsdim + _x]
        else:
            ix = (r'((\1*_ny + _y)*{0} + \2)*lsd{1} + _x'
                  .format(arg.cdims[1], arg.name))

        return '{0}_v[{1}]'.format(arg.name, ix)

    def _render_body(self, body):
        ptns = [r'\b{0}\b', r'\b{0}\[(\d+)\]', r'\b{0}\[(\d+)\]\[(\d+)\]']

        # At single precision suffix all floating point constants by 'f'
        if self.fpdtype == np.float32:
            body = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                          r'\g<0>f', body)

        # Dereference vector arguments
        for va in self.vectargs:
            if va.isview:
                darg = self._deref_arg_view(va)
            else:
                if self.ndim == 1:
                    darg = self._deref_arg_array_1d(va)
                else:
                    darg = self._deref_arg_array_2d(va)

            # Substitute
            body = re.sub(ptns[va.ncdim].format(va.name), darg, body)

        return body
