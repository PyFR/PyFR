# -*- coding: utf-8 -*-

import re

import numpy as np

from pyfr.util import match_paired_paren


class Arg:
    def __init__(self, name, spec, body):
        self.name = name

        specptn = r'''
            (?:(in|inout|out)\s+)?                           # Intent
            (?:(broadcast(?:-row|-col)|mpi|scalar|view)\s+)? # Attrs
            ([A-Za-z_]\w*)                                   # Data type
            ((?:\[\d+\]){0,2})$                              # Dimensions
        '''
        dimsptn = r'(?<=\[)\d+(?=\])'
        usedptn = fr'(?:[^A-Za-z]|^){name}[^A-Za-z0-9]'

        # Parse our specification
        m = re.match(specptn, spec, re.X)
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
        self.isbroadcastr = self.attrs == 'broadcast-row'
        self.isbroadcastc = self.attrs == 'broadcast-col'
        self.ismpi = self.attrs == 'mpi'
        self.isused = bool(re.search(usedptn, body))
        self.isview = self.attrs == 'view'
        self.isscalar = self.attrs == 'scalar'
        self.isvector = not self.isscalar

        # Validation
        if self.attrs.startswith('broadcast') and self.intent != 'in':
            raise ValueError('Broadcast arguments must be of intent in')
        if self.isbroadcastr and self.ncdim != 1:
            raise ValueError('Row broadcasts must have one dimension')
        if self.isbroadcastc and self.ncdim == 1:
            raise ValueError('Column broadcasts must have zero or two dims')
        if self.isscalar and self.dtype != 'fpdtype_t':
            raise ValueError('Scalar arguments must be of type fpdtype_t')


class BaseKernelGenerator:
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

            if va.isview:
                argt.append([np.intp]*(2 + (va.ncdim == 2)))
            elif self.needs_ldim(va):
                argt.append([np.intp, np.int32])
            else:
                argt.append([np.intp])

        # Return
        return self.ndim, argn, argt

    def ldim_size(self, name, *factor):
        return f'ld{name}'

    def needs_ldim(self, arg):
        return self.ndim == 2 or (arg.ncdim > 0 and not arg.ismpi)

    def render(self):
        pass

    def _deref_arg_view(self, arg):
        ptns = [
            '{0}_v[{0}_vix[{1}]]',
            r'{0}_v[{0}_vix[{1}] + SOA_SZ*(\1)]',
            r'{0}_v[{0}_vix[{1}] + {0}_vrstri[{1}]*(\1) + SOA_SZ*(\2)]'
        ]

        return ptns[arg.ncdim].format(arg.name, 'BLK_IDX + X_IDX')

    def _deref_arg_array_1d(self, arg):
        # Vector:
        #   name => name_v[X_IDX + BLK_IDX]
        if arg.ncdim == 0:
            ix = 'X_IDX + BLK_IDX'
        # Tightly packed MPI Vector:
        #   name[\1] => name_v[nx*(\1) + X_IDX + BLK_IDX]
        elif arg.ncdim == 1 and arg.ismpi:
            ix = r'_nx*(\1) + X_IDX + BLK_IDX'
        # Stacked vector:
        #   name[\1] => name_v[ldim*(\1) + X_IDX + BLK_IDX*nv]
        elif arg.ncdim == 1:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*(\1) + X_IDX + BLK_IDX*{arg.cdims[0]}'
        # Doubly stacked MPI vector:
        #   name[\1][\2] => name_v[(nv*(\1) + (\2))*nx + X_IDX + BLK_IDX]
        elif arg.ncdim == 2 and arg.ismpi:
            ix = fr'({arg.cdims[1]}*(\1) + (\2))*_nx + X_IDX + BLK_IDX'
        # Doubly stacked vector:
        #   name[\1][\2] => name_v[ldim*(\1) + X_IDX_AOSOA(\2, nv) +
        #                          BLK_IDX*ns*nv]
        else:
            lx = self.ldim_size(arg.name, arg.cdims[1])
            ix = (fr'{lx}*(\1) + X_IDX_AOSOA(\2, {arg.cdims[1]}) + '
                  f'BLK_IDX*{arg.cdims[0]*arg.cdims[1]}')

        return f'{arg.name}_v[{ix}]'

    def _deref_arg_array_2d(self, arg):
        # Column broadcast matrix with zero dimension:
        #   name => name_v[X_IDX + BLK_IDX]
        if arg.ncdim == 0 and arg.isbroadcastc:
            ix = 'X_IDX + BLK_IDX'
        # Matrix:
        #   name => name_v[ldim*_y + X_IDX + BLK_IDX*ny]
        elif arg.ncdim == 0:
            lx = self.ldim_size(arg.name)
            ix = f'{lx}*_y + X_IDX + BLK_IDX*_ny'
        # Row broadcast matrix
        #   name[\1] => name_v[ldim*_y + \1]
        elif arg.isbroadcastr:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*_y + BCAST_BLK(\1, {lx})'
        # Stacked matrix:
        #   name[\1] => name_v[ldim*_y + X_IDX_AOSOA(\1, nv) + BLK_IDX*nv*ny]
        elif arg.ncdim == 1:
            lx = self.ldim_size(arg.name, arg.cdims[0])
            ix = (fr'{lx}*_y + X_IDX_AOSOA(\1, {arg.cdims[0]}) + '
                  f'BLK_IDX*{arg.cdims[0]}*_ny')
        # Column broadcast matrix
        #   name[\1][\2] => name_v[ldim*\1 + X_IDX_AOSOA(\2, nv) +
        #                          BLK_IDX*ns*nv]
        elif arg.isbroadcastc:
            lx = self.ldim_size(arg.name, arg.cdims[1])
            ix = (fr'{lx}*\1 + X_IDX_AOSOA(\2, {arg.cdims[1]}) + '
                  f'BLK_IDX*{arg.cdims[0]*arg.cdims[1]}')
        # Doubly stacked matrix:
        #   name[\1][\2] => name_v[((\1)*ny + _y)*ldim + X_IDX_AOSOA(\2, nv) +
        #                          BLK_IDX*ns*nv*ny]
        else:
            lx = self.ldim_size(arg.name, arg.cdims[1])
            ix = (fr'((\1)*_ny + _y)*{lx} + '
                  fr'X_IDX_AOSOA(\2, {arg.cdims[1]}) + '
                  f'BLK_IDX*{arg.cdims[0]*arg.cdims[1]}*_ny')

        return f'{arg.name}_v[{ix}]'

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
