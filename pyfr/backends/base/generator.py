from math import prod
import re

import numpy as np

from pyfr.util import match_paired_paren


class Arg:
    def __init__(self, name, spec, body):
        self.name = name

        specptn = r'''
            (?:(in|inout|out)\s+)?                            # Intent
            (?:((?:broadcast-col\s+)?view(?:\(\d+\))?|        # Attrs
                broadcast(?:-row|-col)?|mpi|scalar)\s+)?
            (?:reduce\((min|max|sum)\)\s+)?                   # Reduction
            ([A-Za-z_]\w*)                                    # Data type
            ((?:\[\d+\]){0,2})$                               # Dimensions
        '''
        dimsptn = r'(?<=\[)\d+(?=\])'
        usedptn = fr'(?:[^A-Za-z_]|^){name}\W'

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

        # View stride (for multi-indexed views: view(N))
        m = re.search(r'view\((\d+)\)', self.attrs)
        self.viewstride = int(m[1]) if m else 1

        # Attributes
        self.isbroadcast = self.attrs == 'broadcast'
        self.isbroadcastr = self.attrs == 'broadcast-row'
        self.isbroadcastc = 'broadcast-col' in self.attrs
        self.ismpi = self.attrs == 'mpi'
        self.isview = 'view' in self.attrs
        self.isscalar = self.attrs == 'scalar'
        self.isvector = not self.isscalar
        self.isreduce = bool(self.reduceop)
        self.isused = bool(re.search(usedptn, body))

        # Validation
        if (self.attrs.startswith('broadcast') and
            self.intent != 'in' and not self.isreduce):
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
        if self.isreduce and self.ncdim and not self.isbroadcastc:
            raise ValueError('Non-broadcast-col reduction args must be scalar')
        if self.isscalar and self.dtype not in ('fpdtype_t', 'ixdtype_t'):
            raise ValueError('Scalar arguments must be fpdtype_t or ixdtype_t')


class BaseKernelGenerator:
    def __init__(self, name, ndim, args, body, fpdtype, ixdtype):
        self.name = name
        self.ndim = ndim
        self.fpdtype = fpdtype
        self.ixdtype = ixdtype
        self.fpdtype_max = float(np.finfo(fpdtype).max)

        # Parse and sort our argument list
        sargs = sorted((k, Arg(k, v, body)) for k, v in args.items())

        # Eliminate unused arguments
        sargs = [v for k, v in sargs if v.isused]

        # Break arguments into point-scalars and point-vectors
        self.scalargs = [v for v in sargs if v.isscalar]
        self.vectargs = vargs = [v for v in sargs if v.isvector]

        # Validate 2D argument constraints
        if ndim == 2:
            if any(v.isview and v.intent != 'in' for v in vargs):
                raise ValueError('2D view args must be input-only')
            if any(v.ismpi for v in vargs):
                raise ValueError('2D kernels do not support MPI matrices')

        # Render the main body of our kernel
        body, preamble, epilogue = self._render_body_preamble_epilogue(body)
        self.body, self.preamble, self.epilogue = body, preamble, epilogue

        # Determine the dimensions to be iterated over
        self._dims = ['_nx'] if ndim == 1 else ['_ny', '_nx']

    def argspec(self):
        # Argument names and types
        argn, argt = [], []

        # Dimensions
        argn += self._dims
        argt += [[self.ixdtype]]*self.ndim

        # Scalar args (fpdtype or ixdtype)
        for sa in self.scalargs:
            argn.append(sa.name)
            dtype = self.ixdtype if sa.dtype == 'ixdtype_t' else self.fpdtype
            argt.append([dtype])

        # Vector args
        for va in self.vectargs:
            argn.append(va.name)

            if va.isview:
                match self.ndim, va.ncdim:
                    case 2, _ if va.isbroadcastc:
                        argt.append([np.uintp, np.uintp])
                    case 2, _:
                        argt.append([np.uintp, np.uintp, self.ixdtype])
                    case _, 2:
                        argt.append([np.uintp]*3)
                    case _:
                        argt.append([np.uintp]*2)
            elif self.needs_ldim(va):
                argt.append([np.uintp, self.ixdtype])
            else:
                argt.append([np.uintp])

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
        ncdim = arg.ncdim + (arg.viewstride > 1)
        ptrns = [r'\b{0}\b', r'\b{0}' + bmtch, r'\b{0}' + 2*bmtch]

        return ptrns[ncdim].format(arg.name)

    def _deref_arg(self, arg):
        if arg.isview and self.ndim == 1:
            return self._deref_arg_view_1d(arg)
        elif arg.isview:
            return self._deref_arg_view_2d(arg)
        elif self.ndim == 1:
            return self._deref_arg_array_1d(arg)
        else:
            return self._deref_arg_array_2d(arg)

    def _deref_arg_view_1d(self, arg):
        n = arg.name
        vix = f'{n}_vix[X_IDX]'
        vs = f'{n}_vrstri'

        if arg.viewstride > 1:
            vixs = fr'{n}_vix[X_IDX*{arg.viewstride} + (\1)]'
            return fr'{n}_v[{vixs}]'
        elif arg.ncdim == 0:
            return f'{n}_v[{vix}]'
        elif arg.ncdim == 1:
            return fr'{n}_v[{vix} + SOA_SZ*(\1)]'
        else:
            return fr'{n}_v[{vix} + {vs}[X_IDX]*(\1) + SOA_SZ*(\2)]'

    def _deref_arg_view_2d(self, arg):
        n = arg.name
        vix = f'{n}_vix[X_IDX]'
        vs = f'{n}_vrstri'

        # Broadcast-col view: no _y dependence
        if arg.isbroadcastc and arg.viewstride > 1:
            vixs = fr'{n}_vix[X_IDX*{arg.viewstride} + (\1)]'
            return fr'{n}_v[{vixs}]'
        elif arg.isbroadcastc:
            return f'{n}_v[{vix}]'
        elif arg.ncdim == 0:
            return f'{n}_v[{vix} + {vs}*_y]'
        elif arg.ncdim == 1:
            return fr'{n}_v[{vix} + {vs}*_y + SOA_SZ*(\1)]'
        else:
            return fr'{n}_v[{vix} + {vs}*((\1)*_ny + _y) + SOA_SZ*(\2)]'

    def _deref_arg_array_1d(self, arg):
        # Vector:
        #   name => name_v[X_IDX]
        if arg.ncdim == 0:
            ix = 'X_IDX'
        # 2D broadcast vector
        #   name[\1][\2] => name_v[ldim*(\1) + (\2)]
        elif arg.isbroadcast and arg.ncdim == 2:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*\1 + BCAST_BLK({arg.cdims[0]}, \2, {lx})'
        # Tightly packed MPI Vector:
        #   name[\1] => name_v[nx*(\1) + X_IDX]
        elif arg.ncdim == 1 and arg.ismpi:
            ix = r'_nx*(\1) + X_IDX'
        # Stacked vector:
        #   name[\1] => name_v[ldim*(\1) + X_IDX]
        elif arg.ncdim == 1:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*(\1) + X_IDX'
        # Doubly stacked MPI vector:
        #   name[\1][\2] => name_v[(nv*(\1) + (\2))*nx + X_IDX]
        elif arg.ncdim == 2 and arg.ismpi:
            ix = fr'({arg.cdims[1]}*(\1) + (\2))*_nx + X_IDX'
        # Doubly stacked vector:
        #   name[\1][\2] => name_v[ldim*(\1) + X_IDX_AOSOA(\2, nv)]
        else:
            lx = self.ldim_size(arg.name, arg.cdims[1])
            ix = fr'{lx}*(\1) + X_IDX_AOSOA(\2, {arg.cdims[1]})'

        return f'{arg.name}_v[{ix}]'

    def _deref_arg_array_2d(self, arg):
        # 2D broadcast vector
        #   name[\1][\2] => name_v[ldim*(\1) + (\2)]
        if arg.isbroadcast and arg.ncdim == 2:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*(\1) + BCAST_BLK({arg.cdims[0]}, \2, {lx})'
        # Column broadcast matrix with zero dimension:
        #   name => name_v[X_IDX]
        elif arg.ncdim == 0 and arg.isbroadcastc:
            ix = 'X_IDX'
        # Column broadcast matrix with one dimension
        #   name[\1] => name_v[ldim*(\1) + X_IDX]
        elif arg.ncdim == 1 and arg.isbroadcastc:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*(\1) + X_IDX'
        # Matrix:
        #   name => name_v[ldim*_y + X_IDX]
        elif arg.ncdim == 0:
            lx = self.ldim_size(arg.name)
            ix = f'{lx}*_y + X_IDX'
        # Row broadcast matrix
        #   name[\1] => name_v[ldim*_y + \1]
        elif arg.isbroadcastr:
            lx = self.ldim_size(arg.name)
            ix = fr'{lx}*_y + BCAST_BLK(_ny, \1, {lx})'
        # Stacked matrix:
        #   name[\1] => name_v[ldim*_y + X_IDX_AOSOA(\1, nv)]
        elif arg.ncdim == 1:
            lx = self.ldim_size(arg.name, arg.cdims[0])
            ix = fr'{lx}*_y + X_IDX_AOSOA(\1, {arg.cdims[0]})'
        # Column broadcast matrix
        #   name[\1][\2] => name_v[ldim*\1 + X_IDX_AOSOA(\2, nv)]
        elif arg.isbroadcastc:
            lx = self.ldim_size(arg.name, arg.cdims[1])
            ix = fr'{lx}*(\1) + X_IDX_AOSOA(\2, {arg.cdims[1]})'
        # Doubly stacked matrix:
        #   name[\1][\2] => name_v[((\1)*ny + _y)*ldim + X_IDX_AOSOA(\2, nv)]
        else:
            lx = self.ldim_size(arg.name, arg.cdims[1])
            ix = fr'((\1)*_ny + _y)*{lx} + X_IDX_AOSOA(\2, {arg.cdims[1]})'

        return f'{arg.name}_v[{ix}]'

    def _render_body(self, body):
        # At single precision suffix all floating point constants by 'f'
        if self.fpdtype == np.float32:
            body = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                          r'\g<0>f', body)

        # Track 2D broadcast-col reductions for deferred writeback
        self._reduce_args = []

        # Dereference vector arguments
        for va in self.vectargs:
            subp = self._match_arg(va)
            darg = self._deref_arg(va)

            # Regular
            if not va.isreduce:
                body = re.sub(subp, darg, body)
            # Reduction
            else:
                body = self._render_reduce(va, body, subp, darg)

        return body

    def _reduce_ident(self, reduceop):
        m = self.fpdtype_max
        return {'sum': 0, 'min': m, 'max': -m}[reduceop]

    _reduce_fn = {'min': 'fmin', 'max': 'fmax'}

    def _accum_expr(self, reduceop, dst, src):
        if reduceop == 'sum':
            return f'{dst} += {src};'
        else:
            rfn = self._reduce_fn[reduceop]
            return f'{dst} = {rfn}({dst}, {src});'

    def _render_reduce(self, va, body, subp, darg):
        n = prod(va.cdims) if va.cdims else 1

        # 2D broadcast-col: per-iteration local + register accumulator
        if va.isbroadcastc:
            rname = f'_ra_{va.name}'
            ident = self._reduce_ident(va.reduceop)
            self._reduce_args.append(
                (rname, darg, ident, va.reduceop, n, va.ncdim)
            )

            if va.ncdim:
                decl = f'fpdtype_t {va.name}[{n}];'
                accum = '\n'.join(
                    self._accum_expr(va.reduceop, f'{rname}[{j}]',
                                     f'{va.name}[{j}]')
                    for j in range(n))
            else:
                decl = f'fpdtype_t {va.name};'
                accum = self._accum_expr(va.reduceop, rname, va.name)

            return f'{decl}\n{body}\n{accum}'
        # 1D: local variable + immediate atomic writeback
        else:
            afun = f'atomic_{va.reduceop}_fpdtype'
            vs = va.viewstride

            if vs > 1:
                body = f'fpdtype_t {va.name}[{vs}];\n{body}'
                for i in range(vs):
                    gval = re.sub(subp, darg, f'{va.name}[{i}]')
                    body += f'{afun}(&{gval}, {va.name}[{i}]);\n'
            else:
                body = f'fpdtype_t {va.name};\n{body}'
                body += f'{afun}(&{darg}, {va.name});\n'

            return body

    def _render_body_preamble_epilogue(self, body):
        return self._render_body(body), '', ''


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
            blk_y, lid_y = self.block2d[1], self._lid[1]
            self.preamble += f'''
                ixdtype_t _ysize = (_ny + {blk_y - 1}) / {blk_y};
                ixdtype_t _ystart = {lid_y}*_ysize;
                ixdtype_t _yend = min(_ny, _ystart + _ysize);
                for (ixdtype_t _y = _ystart; _x < _nx && _y < _yend; _y++)'''

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

    def _render_body_preamble_epilogue(self, body):
        preamble = ''

        # For 2D kernels, preload data from column broadcast arrays
        # into shared memory to guarantee reuse
        if self.ndim == 2:
            preload, usedb = [], 0

            for va in self.vectargs:
                if va.isbroadcastc and va.ncdim >= 1 and not va.isreduce:
                    # Preload the argument into shared memory
                    sname, lcode, nbytes = self._preload_arg(va)

                    # Limit ourselves to 32KiB of shared state
                    if usedb + nbytes > 32*1024:
                        continue

                    preload.append(lcode)
                    usedb += nbytes

                    if va.ncdim == 1:
                        subs = fr'{sname}[\1][{self._lid[0]}]'
                    else:
                        subs = fr'{sname}[\1][\2][{self._lid[0]}]'

                    # Substitute all uses of the argument in the body
                    body = re.sub(self._match_arg(va), subs, body)

            if preload:
                preload.append(f'{self._shared_sync};')
                preamble = '\n'.join(preload)

        body = self._render_body(body)

        # 2D reduction: accumulator decls + shared-memory epilogue
        rpre, epilogue = self._render_reduce_2d()
        preamble += '\n' + rpre

        return body, preamble, epilogue

    def _render_reduce_2d(self):
        if not self._reduce_args:
            return '', ''

        bx = self.block2d[0]
        lx, ly = self._lid

        # Preamble: register accumulators initialised to identity
        preamble_parts = []
        for rn, _, ident, _, n, ncdim in self._reduce_args:
            if ncdim:
                inits = ' '.join(f'{rn}[{j}] = {ident};' for j in range(n))
                preamble_parts.append(f'fpdtype_t {rn}[{n}]; {inits}')
            else:
                preamble_parts.append(f'fpdtype_t {rn} = {ident};')
        preamble = '\n'.join(preamble_parts)

        # Epilogue: shared-memory reduce across y-threads, then
        # write result to global (single shared array, reused)
        epilogue = f'{self._shared_prfx} fpdtype_t _rs[{bx}];'
        for rname, darg, ident, reduceop, n, ncdim in self._reduce_args:
            afn = f'atomic_{reduceop}_fpdtype'

            # Build (register-name, global-name) pairs for each component
            if ncdim:
                pairs = [(f'{rname}[{j}]', darg.replace('\\1', str(j)))
                         for j in range(n)]
            else:
                pairs = [(rname, darg)]

            for src, dst in pairs:
                epilogue += f'''
                    if ({ly} == 0) _rs[{lx}] = {ident};
                    {self._shared_sync};
                    if (_x < _nx) {afn}(&_rs[{lx}], {src});
                    {self._shared_sync};
                    if ({ly} == 0 && _x < _nx) {dst} = _rs[{lx}];
                '''

        return preamble, epilogue

    def render(self):
        spec = self._render_spec()

        return f'''{spec}
            {{
                ixdtype_t _x = {self._gid};
                #define X_IDX (_x)
                #define X_IDX_AOSOA(v, nv) SOA_IX(X_IDX, v, nv)
                #define BCAST_BLK(r, c, ld)  c
                {self.preamble}
                {{
                    {self.body}
                }}
                {self.epilogue}
                #undef X_IDX
                #undef X_IDX_AOSOA
                #undef BCAST_BLK
            }}'''
