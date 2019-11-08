# -*- coding: utf-8 -*-

import re

import numpy as np

from pyfr.util import match_paired_paren


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
        if self.isscalar and self.dtype != 'fpdtype_t':
            raise ValueError('Scalar arguments must be of type fpdtype_t')


class BaseKernelGenerator(object):
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
                argt.append([np.intp]*(2 + (va.ncdim == 2)))
            # Broadcast vector
            elif va.isbroadcast:
                argt.append([np.intp])
            # Non-stacked vector or MPI type
            elif self.ndim == 1 and (va.ncdim == 0 or va.ismpi):
                argt.append([np.intp])
            # Stacked vector/matrix/stacked matrix
            else:
                argt.append([np.intp, np.int32])

        # Return
        return self.ndim, argn, argt

    def needs_ldim(self, arg):
        if arg.isbroadcast:
            return ((self.ndim == 1 and arg.ncdim > 1) or
                    (self.ndim == 2 and arg.ncdim > 0))
        else:
            return self.ndim == 2 or (arg.ncdim > 0 and not arg.ismpi)

    def render(self):
        pass

    def _deref_arg_view(self, arg):
        ptns = [
            '{0}_v[{0}_vix[X_IDX]]',
            r'{0}_v[{0}_vix[X_IDX] + SOA_SZ*(\1)]',
            r'{0}_v[{0}_vix[X_IDX] + {0}_vrstri[X_IDX]*(\1) + SOA_SZ*(\2)]'
        ]

        return ptns[arg.ncdim].format(arg.name)

    def _deref_arg_array_1d(self, arg):
        # Leading dimension
        ldim = 'ld' + arg.name if not arg.ismpi else '_nx'

        # Broadcast vector
        #   name[\1] => name_v[\1]
        if arg.isbroadcast:
            ix = r'\1'
        # Vector:
        #   name => name_v[X_IDX]
        elif arg.ncdim == 0:
            ix = 'X_IDX'
        # Stacked vector:
        #   name[\1] => name_v[ldim*(\1) + X_IDX]
        elif arg.ncdim == 1:
            ix = r'{0}*(\1) + X_IDX'.format(ldim)
        # Doubly stacked MPI vector:
        #   name[\1][\2] => name_v[(nv*(\1) + (\2))*ldim + X_IDX]
        elif arg.ismpi:
            ix = r'({0}*(\1) + (\2))*{1} + X_IDX'.format(arg.cdims[1], ldim)
        # Doubly stacked vector:
        #   name[\1][\2] => name_v[ldim*(\1) + X_IDX_AOSOA(\2, nv)]
        else:
            ix = (r'ld{0}*(\1) + X_IDX_AOSOA(\2, {1})'
                   .format(arg.name, arg.cdims[1]))

        return '{0}_v[{1}]'.format(arg.name, ix)

    def _deref_arg_array_2d(self, arg):
        # Broadcast vector:
        #   name => name_v[X_IDX]
        if arg.isbroadcast:
            ix = 'X_IDX'
        # Matrix:
        #   name => name_v[ldim*_y + X_IDX]
        elif arg.ncdim == 0:
            ix = 'ld{0}*_y + X_IDX'.format(arg.name)
        # Stacked matrix:
        #   name[\1] => name_v[ldim*_y + X_IDX_AOSOA(\1, nv)]
        elif arg.ncdim == 1:
            ix = (r'ld{0}*_y + X_IDX_AOSOA(\1, {1})'
                   .format(arg.name, arg.cdims[0]))
        # Doubly stacked matrix:
        #   name[\1][\2] => name_v[((\1)*ny + _y)*ldim + X_IDX_AOSOA(\2, nv)]
        else:
            ix = (r'((\1)*_ny + _y)*ld{0} + X_IDX_AOSOA(\2, {1})'
                   .format(arg.name, arg.cdims[1]))

        return '{0}_v[{1}]'.format(arg.name, ix)

    def _render_body(self, body):
        bmch = r'\[({0})\]'.format(match_paired_paren('[]'))
        ptns = [r'\b{0}\b', r'\b{0}' + bmch, r'\b{0}' + 2*bmch]

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
