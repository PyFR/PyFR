from math import prod

import numpy as np

from pyfr.backends.base.generator.base import BaseKernelGenerator
from pyfr.dsl.nodes import Assign, ExprStmt, Float, Index, Program, Var
from pyfr.dsl.rewriter import Rewriter, parse_expr, rename_vars


class BaseGPUKernelGenerator(BaseKernelGenerator):
    # Block sizes for 1D and 2D kernels, respectively
    block1d = None
    block2d = None

    # Expressions for local x/y id's and global x id
    _lid = None
    _gid = None

    # Prefix for variables in shared memory
    _shared_prfx = None

    # Declaration of a pointer argument, const-qualified where read-only
    _ptr_arg = '{c}{t}* {n}'

    # Expression for synchronising shared memory
    _shared_sync = None

    # Prototype expressions for preloading 1D and 2D array arguments
    _preload_protos = {1: parse_expr('@S[@I0][@LX]'),
                       2: parse_expr('@S[@I0][@I1][@LX]')}

    # Lowerings for the @-intrinsics produced by the dereference rules
    _xidx = '_x'
    _aosoa = '(_x / @soasz*@nv + @v)*@soasz + _x % @soasz'
    _bcast = '@c'

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

    def _kargs(self):
        # Arguments of the kernel as (pointer, const, type, name)
        kargs = [(False, False, 'ixdtype_t', d) for d in self._dims]
        kargs += [(False, False, sa.dtype, sa.name) for sa in self.scalargs]

        for va in self.vectargs:
            n, ro = va.name, va.intent == 'in'
            kargs.append((True, ro, va.dtype, f'{n}_v'))

            # Views
            if va.isview:
                kargs.append((True, True, 'ixdtype_t', f'{n}_vix'))

                if self.ndim == 2 and not va.isbroadcastc:
                    kargs.append((False, False, 'ixdtype_t', f'{n}_vrstri'))
                elif va.ncdim == 2 and va.cdims[0] > 1:
                    kargs.append((True, True, 'ixdtype_t', f'{n}_vrstri'))
            # Arrays
            elif self.needs_ldim(va):
                kargs.append((False, False, 'ixdtype_t', f'ld{n}'))

        return kargs

    def _karg(self, ptr, const, t, n):
        # Declare an argument as the backend spells it
        if ptr:
            return self._ptr_arg.format(c='const ' if const else '', t=t, n=n)
        else:
            return f'{t} {n}'

    def ldim_size(self, name, factor=1):
        return Var(f'ld{name}')

    def needs_ldim(self, arg):
        return self.ndim == 2 or (arg.ncdim > 0 and not arg.ismpi)

    def _preload_rule(self, arg, sname):
        pname, pncdim = arg.name, arg.ncdim
        proto = self._preload_protos[pncdim]

        def preload_sub(n, ix, a):
            if n == pname and a.ncdim == pncdim and len(ix) == pncdim:
                subs = {f'I{i}': idx for i, idx in enumerate(ix)}
                subs['S'] = Var(sname)
                subs['LX'] = Var(str(self._lid[0]))
                return rename_vars(proto, subs)
            else:
                return None

        return preload_sub

    def _preload_arg(self, arg, codegen):
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
        if arg.ncdim == 1:
            lhs = f'{sname}[_i][{lx}]'
            rhs = self._generate(codegen, self._deref_arg(arg, '_i'))
        else:
            dim1 = f'(_i / {arg.cdims[1]})'
            dim2 = f'(_i % {arg.cdims[1]})'

            lhs = f'{sname}[{dim1}][{dim2}][{lx}]'
            rhs = self._generate(codegen, self._deref_arg(arg, dim1, dim2))

        # Declare the shared array
        lcode = f'{sprfx} {arg.dtype} {sname}{arg.cdimstr}[{bx}];'

        # Emit the for loop to populate the array
        lcode += f'''
            for (int _i = {ly}; _x < _nx && _i < {n}; _i += {by})
                {lhs} = {rhs};'''

        return sname, lcode, nbytes

    def _render_body_preamble_epilogue(self):
        # Track 2D broadcast-col reductions for deferred writeback
        self._reduce_args = []

        codegen = self.codegen_cls()
        preamble = ''
        preload_rules = []

        # For 2D kernels, preload data from column broadcast arrays
        # into shared memory to guarantee reuse
        if self.ndim == 2:
            preload, usedb = [], 0

            for va in self.vectargs:
                if va.isbroadcastc and va.ncdim >= 1 and not va.isreduce:
                    # Preload the argument into shared memory
                    sname, lcode, nbytes = self._preload_arg(va, codegen)

                    # Limit ourselves to 32KiB of shared state
                    if usedb + nbytes > 32*1024:
                        continue

                    preload.append(lcode)
                    usedb += nbytes

                    # Redirect uses of the argument to the shared array
                    preload_rules.append(self._preload_rule(va, sname))

            if preload:
                preload.append(f'{self._shared_sync};')
                preamble = '\n'.join(preload)

        # Substitute the rules through the body; preloads take priority
        args = {va.name: va for va in self.vectargs if not va.isreduce}
        rules = preload_rules + self._deref_rules
        ast = Rewriter(rules).transform(self._ast, args)

        # Render the body
        body = self._render_body(ast, codegen)

        # 2D reduction: accumulator decls + shared-memory epilogue
        rpre, epilogue = self._render_reduce_2d(codegen)
        preamble += '\n' + rpre

        return body, preamble, epilogue

    def _render_reduce_2d(self, codegen):
        if not self._reduce_args:
            return '', ''

        bx = self.block2d[0]
        lx, ly = self._lid

        # Preamble: register accumulators initialised to identity
        stmts = []
        for rdecl, pairs, ident, _ in self._reduce_args:
            iexpr = Float(float(ident))
            stmts.append(rdecl)
            stmts += [ExprStmt(Assign(acc, iexpr)) for acc, _ in pairs]
        preamble = self._generate(codegen, Program(stmts))

        # Epilogue: shared-memory reduce across y-threads, then
        # write result to global (single shared array, reused)
        rs = Index(Var('_rs'), Var(str(lx)))
        epilogue = f'{self._shared_prfx} fpdtype_t _rs[{bx}];'
        for _, pairs, ident, reduceop in self._reduce_args:
            for src, dst in pairs:
                atom = self._generate(codegen,
                                      self._atomic_stmt(reduceop, rs, src))
                wb = self._generate(codegen, ExprStmt(Assign(dst, rs)))
                epilogue += f'''
                    if ({ly} == 0) _rs[{lx}] = {ident};
                    {self._shared_sync};
                    if (_x < _nx) {atom}
                    {self._shared_sync};
                    if ({ly} == 0 && _x < _nx) {wb}
                '''

        return preamble, epilogue

    def render(self):
        spec = self._render_spec()

        return f'''{self.helpers}
            {spec}
            {{
                ixdtype_t _x = {self._gid};
                {self.preamble}
                {{
                    {self.body}
                }}
                {self.epilogue}
            }}'''
