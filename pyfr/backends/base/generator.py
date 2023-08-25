from math import prod
import re

import numpy as np

from pyfr.util import match_paired_paren, ndrange


class Arg:
    def __init__(self, name, spec, body):
        self.name = name

        specptn = r'''
            (?:(in|inout|out)\s+)?                            # Intent
            (?:(broadcast(?:-row|-col)?|mpi|scalar|view)\s+)? # Attrs
            (?:reduce\((min)\)\s+)?                           # Reduction
            ([A-Za-z_]\w*)                                    # Data type
            ((?:\[\d+\]){0,2})$                               # Dimensions
        '''
        dimsptn = r'(?<=\[)\d+(?=\])'
        usedptn = fr'(?:[^A-Za-z_]|^){name}[^A-Za-z0-9]'

        # Parse our specification
        m = re.match(specptn, spec, re.X)
        if not m:
            raise ValueError('Invalid argument specification')

        g = m.groups()

        # Properties
        self.intent = g[0] or 'in'
        self.attrs = g[1] or ''
        self.reduceop = g[2]
        self.dtype = g[3]
        self.cdimstr = g[4]

        # Dimension
        self.cdims = [int(d) for d in re.findall(dimsptn, g[4])]
        self.ncdim = len(self.cdims)

        # Attributes
        self.isbroadcast = self.attrs == 'broadcast'
        self.isbroadcastr = self.attrs == 'broadcast-row'
        self.isbroadcastc = self.attrs == 'broadcast-col'
        self.ismpi = self.attrs == 'mpi'
        self.isview = self.attrs == 'view'
        self.isscalar = self.attrs == 'scalar'
        self.isvector = not self.isscalar
        self.isreduce = bool(self.reduceop)
        self.isused = bool(re.search(usedptn, body))

        # Validation
        if self.attrs.startswith('broadcast') and self.intent != 'in':
            raise ValueError('Broadcast arguments must be of intent in')
        if self.isbroadcast and self.ncdim != 2:
            raise ValueError('Broadcasts must have two dimensions')
        if self.isbroadcastr and self.ncdim != 1:
            raise ValueError('Row broadcasts must have one dimension')
        if self.isreduce and self.intent != 'out':
            raise ValueError('Reduction arguments must be of intent out')
        if self.isreduce and self.dtype != 'fpdtype_t':
            raise ValueError('Reduction arguments must be of type fpdtype_t')
        if self.isreduce and self.isscalar:
            raise ValueError('Scalar arguments can not be reduced')
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
        self.body, self.preamble = self._render_body_preamble(body)

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

    def ldim_size(self, name, factor=1):
        pass

    def needs_ldim(self, arg):
        pass

    def render(self):
        pass

    def _match_arg(self, arg):
        bmtch = r'\[({0})\]'.format(match_paired_paren('[]'))
        ptrns = [r'\b{0}\b', r'\b{0}' + bmtch, r'\b{0}' + 2*bmtch]

        return ptrns[arg.ncdim].format(arg.name)

    def _deref_arg(self, arg):
        if arg.isview:
            return self._deref_arg_view(arg)
        elif self.ndim == 1:
            return self._deref_arg_array_1d(arg)
        else:
            return self._deref_arg_array_2d(arg)

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
        # 2D broadcast vector
        #   name[\1][\2] => name_v[ldim*(\1) + (\2)]
        elif arg.isbroadcast and arg.ncdim == 2:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*\1 + BCAST_BLK({arg.cdims[0]}, \2, {lx})'
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
        # 2D broadcast vector
        #   name[\1][\2] => name_v[ldim*(\1) + (\2)]
        if arg.isbroadcast and arg.ncdim == 2:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*(\1) + BCAST_BLK({arg.cdims[0]}, \2, {lx})'
        # Column broadcast matrix with zero dimension:
        #   name => name_v[X_IDX + BLK_IDX]
        elif arg.ncdim == 0 and arg.isbroadcastc:
            ix = 'X_IDX + BLK_IDX'
        # Column broadcast matrix with one dimension
        #   name[\1] => name_v[ldim*(\1) + X_IDX + BLK_IDX*ns]
        elif arg.ncdim == 1 and arg.isbroadcastc:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*(\1) + X_IDX + BLK_IDX*{arg.cdims[0]}'
        # Matrix:
        #   name => name_v[ldim*_y + X_IDX + BLK_IDX*ny]
        elif arg.ncdim == 0:
            lx = self.ldim_size(arg.name)
            ix = f'{lx}*_y + X_IDX + BLK_IDX*_ny'
        # Row broadcast matrix
        #   name[\1] => name_v[ldim*_y + \1]
        elif arg.isbroadcastr:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*_y + BCAST_BLK(_ny, \1, {lx})'
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
            ix = (fr'{lx}*(\1) + X_IDX_AOSOA(\2, {arg.cdims[1]}) + '
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
        # At single precision suffix all floating point constants by 'f'
        if self.fpdtype == np.float32:
            body = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                          r'\g<0>f', body)

        # Dereference vector arguments
        for va in self.vectargs:
            subp = self._match_arg(va)
            darg = self._deref_arg(va)

            # Regular
            if not va.isreduce:
                body = re.sub(subp, darg, body)
            # Reduction
            else:
                afun = f'atomic_{va.reduceop}_fpdtype'
                body = f'fpdtype_t {va.name}{va.cdimstr};\n{body}'

                if va.ncdim == 0:
                    body += f'{afun}(&{darg}, {va.name});\n'
                else:
                    for ij in ndrange(*va.cdims):
                        lval = va.name + ''.join(f'[{i}]' for i in ij)
                        gval = re.sub(subp, darg, lval)

                        body += f'{afun}(&{gval}, {lval});\n'

        return body

    def _render_body_preamble(self, body):
        return self._render_body(body), ''


class BaseGPUKernelGenerator(BaseKernelGenerator):
    # Block sizes for 1D and 2D kernels, respectively
    block1d = None
    block2d = None

    # Expressions for local x/y id's and global x id
    _lid = None
    _gid = None

    # Prefix for variables in shared memory
    _shared_prfx = None

    # Expression for synchronising shared memory
    _shared_sync = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self.preamble += 'if (_x < _nx)'
        else:
            self.preamble += f'''
                int _ysize = (_ny + {self.block2d[1] - 1}) / {self.block2d[1]};
                int _ystart = {self._lid[1]}*_ysize;
                int _yend = min(_ny, _ystart + _ysize);
                for (int _y = _ystart; _x < _nx && _y < _yend; _y++)'''

    def ldim_size(self, name, factor=1):
        return f'ld{name}'

    def needs_ldim(self, arg):
        return self.ndim == 2 or (arg.ncdim > 0 and not arg.ismpi)

    def _preload_arg(self, arg):
        bx, by = self.block2d[:2]
        lx, ly = self._lid
        sprfx = self._shared_prfx

        sname = f'{arg.name}_s'

        # Determine the total number of elements in the array
        n = prod(arg.cdims)

        if arg.dtype == 'fpdtype_t':
            itemsize = np.dtype(self.fpdtype).itemsize
        else:
            itemsize = 4

        # Tally up the number of bytes required for the shared array
        nbytes = n*bx*itemsize

        # Dereference the argument
        rhs = self._deref_arg(arg)

        if arg.ncdim == 1:
            lhs = f'{sname}[_i][{lx}]'
            rhs = rhs.replace(r'\1', '_i')
        else:
            dim1 = f'(_i / {arg.cdims[1]})'
            dim2 = f'(_i % {arg.cdims[1]})'

            lhs = f'{sname}[{dim1}][{dim2}][{lx}]'
            rhs = rhs.replace(r'\1', dim1).replace(r'\2', dim2)

        # Declare the shared array
        lcode = f'{sprfx} {arg.dtype} {sname}{arg.cdimstr}[{bx}];'

        # Emit the for loop to populate the array
        lcode += f'''
            for (int _i = {ly}; _x < _nx && _i < {n}; _i += {by})
                {lhs} = {rhs};'''

        return sname, lcode, nbytes

    def _render_body_preamble(self, body):
        preamble = []

        # For 2D kernels, preload data from column broadcast arrays
        # into shared memory to guarantee reuse
        if self.ndim == 2:
            usedb = 0

            for va in self.vectargs:
                if va.isbroadcastc and va.ncdim >= 1:
                    # Preload the argument into shared memory
                    sname, lcode, nbytes = self._preload_arg(va)

                    # Limit ourselves to 32KiB of shared state
                    if usedb + nbytes > 32*1024:
                        continue

                    preamble.append(lcode)
                    usedb += nbytes

                    if va.ncdim == 1:
                        subs = fr'{sname}[\1][{self._lid[0]}]'
                    else:
                        subs = fr'{sname}[\1][\2][{self._lid[0]}]'

                    # Substitute all uses of the argument in the body
                    body = re.sub(self._match_arg(va), subs, body)

        # If we performed any preloading then emit a shared memory barrier
        if preamble:
            preamble.append(f'{self._shared_sync};')

        return self._render_body(body), '\n'.join(preamble)

    def render(self):
        spec = self._render_spec()

        return f'''{spec}
            {{
                int _x = {self._gid};
                #define X_IDX (_x)
                #define X_IDX_AOSOA(v, nv) SOA_IX(X_IDX, v, nv)
                #define BLK_IDX 0
                #define BCAST_BLK(r, c, ld)  c
                {self.preamble}
                {{
                    {self.body}
                }}
                #undef X_IDX
                #undef X_IDX_AOSOA
                #undef BLK_IDX
                #undef BCAST_BLK
            }}'''
