from math import prod

from pyfr.backends.base.generator import BaseKernelGenerator
from pyfr.dsl.nodes import (Binary, DslVar, Index, Int, Program, Var,
                            VarDecl, map_ast)


class OpenMPKernelGenerator(BaseKernelGenerator):
    # Math functions we lower to helper functions in all kernels
    lower_fns = frozenset({'exp'})

    # Lowerings for the $-intrinsics produced by the dereference rules
    _xidx = '_xi + _xj'
    _aosoa = '(_xi / $soasz*$nv + $v)*$soasz + _xj'
    _bcast = '$c % $ld + ($c / $ld)*$ld*$r'

    def _render_body_preamble_epilogue(self):
        self._staged_reduces = []
        self._bcol_reduces = []
        return super()._render_body_preamble_epilogue()

    def _finalise_ast(self, ast):
        # Redirect 1D view reductions through their staging arrays
        for va in self.vectargs:
            if va.isreduce and not va.isbroadcastc:
                self._staged_reduces.append(va)
                ast = self._stage_reduce(ast, va)

        return ast

    def _stage_reduce(self, ast, va):
        rv, xidx = Var(f'_rv_{va.name}'), DslVar('xidx')

        def stage(node):
            node = map_ast(node, stage)
            match node:
                # Rewrite name into _rv_name[$xidx]
                case Var(name) if name == va.name and va.viewstride == 1:
                    return Index(rv, xidx)
                # Rewrite name[i] into _rv_name[i][$xidx]
                case Index(Var(name), ix) if name == va.name:
                    return Index(Index(rv, ix), xidx)
                case _:
                    return node

        return stage(ast)

    def _render_reduce(self, va, body, codegen):
        # 2D broadcast-col: per-iteration local + direct accumulation
        if va.isbroadcastc:
            self._bcol_reduces.append(va)
            n = prod(va.cdims) if va.cdims else 1

            if va.ncdim:
                # Declare a local accumulator fpdtype_t name[n]
                decl = VarDecl(False, 'fpdtype_t', va.name, [Int(n)])
                # Emit name_v[...] = op(name_v[...], name[j]) per element
                accum = [self._accum_stmt(va.reduceop, self._deref_arg(va, j),
                                          Index(Var(va.name), Int(j)))
                         for j in range(n)]
            else:
                # Declare a local accumulator fpdtype_t name
                decl = VarDecl(False, 'fpdtype_t', va.name)
                # Emit name_v[...] = op(name_v[...], name)
                accum = [self._accum_stmt(va.reduceop, self._deref_arg(va),
                                          Var(va.name))]

            dstr = self._generate(codegen, decl)
            astr = self._generate(codegen, Program(accum))
            return f'{dstr}\n{body}\n{astr}'
        # 1D view reductions have already been staged by _finalise_ast
        else:
            return body

    def _render_staging(self):
        # Generate staging array declarations and atomic writeback code
        codegen = self.codegen_cls()
        decls, atoms = [], []

        # Index _xi + _xj of the current lane in the staging arrays
        ix = Binary('+', Var('_xi'), Var('_xj'))

        for va in self._staged_reduces:
            name = va.name
            rv, vix = Var(f'_rv_{name}'), Var(f'{name}_vix')
            gv = Var(f'{name}_v')
            vs = va.viewstride

            if vs > 1:
                # Declare the staging array fpdtype_t _rv_name[vs][BLK_SZ]
                sizes = [Int(vs), Var('BLK_SZ')]
                decls.append(VarDecl(False, 'fpdtype_t', rv.name, sizes))
                for i in range(vs):
                    # Flush _rv_name[i][ix] into name_v[name_vix[vs*ix + i]]
                    vidx = Binary('+', Binary('*', Int(vs), ix), Int(i))
                    dst = Index(gv, Index(vix, vidx))
                    src = Index(Index(rv, Int(i)), ix)
                    atoms.append(self._atomic_stmt(va.reduceop, dst, src))
            else:
                # Declare the staging array fpdtype_t _rv_name[BLK_SZ]
                sizes = [Var('BLK_SZ')]
                decls.append(VarDecl(False, 'fpdtype_t', rv.name, sizes))
                # Flush _rv_name[ix] into name_v[name_vix[ix]]
                dst, src = Index(gv, Index(vix, ix)), Index(rv, ix)
                atoms.append(self._atomic_stmt(va.reduceop, dst, src))

        dstr = self._generate(codegen, Program(decls))
        astr = self._generate(codegen, Program(atoms))
        return dstr, astr

    def _render_bcol_reduce_init(self):
        # Identity-initialise broadcast-col reduce outputs
        lines = []
        for va in self._bcol_reduces:
            n = prod(va.cdims) if va.cdims else 1
            ident = self._reduce_ident(va.reduceop)
            lines.append(f'for (int _i = 0; _i < BLK_SZ*{n}; _i++) '
                         f'{va.name}_v[_i] = {ident};')

        return '\n'.join(lines)

    def render(self):
        kargdefn, kargassn = self._render_args('args')
        rdecls, rflush = self._render_staging()

        # View reduce: atomic flush after each SIMD chunk
        if rflush:
            rflush_core = (f'for (int _xj = 0; _xj < {self.soasz}; _xj++) '
                           f'{{ {rflush} }}')
            rflush_clean = (f'for (int _xj = 0; _xj < _xjn; _xj++) '
                            f'{{ {rflush} }}')
        else:
            rflush_core = rflush_clean = ''

        # Broadcast-col reduce identity init
        rinit = self._render_bcol_reduce_init()

        if self.ndim == 1:
            core = f'''
                for (int _xi = 0; _xi < BLK_SZ; _xi += {self.soasz})
                {{
                    #pragma omp simd
                    for (int _xj = 0; _xj < {self.soasz}; _xj++)
                    {{
                        {self.body}
                    }}
                    {rflush_core}
                }}'''
            clean = f'''
                int _rem = _nx % BLK_SZ;
                for (int _xi = 0; _xi < _rem; _xi += {self.soasz})
                {{
                    int _xjn = min({self.soasz}, _rem - _xi);
                    #pragma omp simd
                    for (int _xj = 0; _xj < _xjn; _xj++)
                    {{
                        {self.body}
                    }}
                    {rflush_clean}
                }}'''
        else:
            core = f'''
                {rinit}
                for (ixdtype_t _y = 0; _y < _ny; _y++)
                {{
                    for (int _xi = 0; _xi < BLK_SZ; _xi += {self.soasz})
                    {{
                        #pragma omp simd
                        for (int _xj = 0; _xj < {self.soasz}; _xj++)
                        {{
                            {self.body}
                        }}
                        {rflush_core}
                    }}
                }}'''
            clean = f'''
                {rinit}
                int _rem = _nx % BLK_SZ;
                for (ixdtype_t _y = 0; _y < _ny; _y++)
                {{
                    for (int _xi = 0; _xi < _rem; _xi += {self.soasz})
                    {{
                        int _xjn = min({self.soasz}, _rem - _xi);
                        #pragma omp simd
                        for (int _xj = 0; _xj < _xjn; _xj++)
                        {{
                            {self.body}
                        }}
                        {rflush_clean}
                    }}
                }}'''

        result = f'''
            {self.helpers}
            struct {self.name}_kargs {{ {kargdefn}; }};
            void {self.name}(ixdtype_t _ib,
                             const struct {self.name}_kargs *args,
                             int _disp_mask)
            {{
                {kargassn};
                {rdecls}
                if (_nx - _ib*BLK_SZ >= BLK_SZ)
                {{
                    {core}
                }}
                else
                {{
                    {clean}
                }}
            }}'''

        return result

    def ldim_size(self, name, factor=1):
        if factor > 1:
            return Binary('*', Int(factor), Var('BLK_SZ'))
        else:
            return Var('BLK_SZ')

    def needs_ldim(self, arg):
        return False

    def _displace_arg(self, arg):
        if arg.isview:
            return None
        elif self.ndim == 1:
            # Vector
            if arg.ncdim == 0 or arg.ismpi:
                return '_ib*BLK_SZ'
            # 2D broadcast vector
            elif arg.isbroadcast:
                return None
            # Stacked vector:
            else:
                return f'_ib*BLK_SZ*{prod(arg.cdims)}'
        else:
            # 2D broadcast vector or row broadcast matrix
            if arg.isbroadcast or arg.isbroadcastr:
                return None
            # Column broadcast matrix
            elif arg.isbroadcastc:
                return f'_ib*BLK_SZ*{prod(arg.cdims)}'
            # Matrix
            else:
                return f'_ib*BLK_SZ*{prod(arg.cdims)}*_ny'

    def _render_args(self, argn):
        # We first need the argument list; starting with the dimensions
        kargs = [('ixdtype_t', d, None, None) for d in self._dims]

        # Now add any scalar arguments
        kargs.extend((sa.dtype, sa.name, None, None) for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            da = self._displace_arg(va)
            mi = len(kargs) if da else None

            if va.intent == 'in':
                kargs.append((f'const {va.dtype}*', f'{va.name}_v', da, mi))
            else:
                kargs.append((f'{va.dtype}*', f'{va.name}_v', da, mi))

            # Views
            if va.isview:
                vix_disp = f'_ib*BLK_SZ*{va.viewstride}'
                kargs.append(('const ixdtype_t*', f'{va.name}_vix', vix_disp,
                              None))

                if self.ndim == 2 and not va.isbroadcastc:
                    kargs.append(('ixdtype_t', f'{va.name}_vrstri', None,
                                  None))
                elif va.ncdim == 2 and va.cdims[0] > 1:
                    kargs.append(('const ixdtype_t*', f'{va.name}_vrstri',
                                  '_ib*BLK_SZ', None))

        # Argument definitions and assignments
        kargdefn, kargassn = [], []
        for dtype, name, disp, midx in kargs:
            assn = f'{dtype} {name} = {argn}->{name}'

            # Handle displacement and potential masking thereof
            if disp and midx is not None:
                assn += f' + ((_disp_mask & {1 << midx}) ? 0 : {disp})'
            elif disp:
                assn += f' + {disp}'

            kargdefn.append(f'{dtype} {name}')
            kargassn.append(assn)

        return ';\n'.join(kargdefn), ';\n'.join(kargassn)
