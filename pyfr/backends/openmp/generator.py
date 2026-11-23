from math import prod

from pyfr.backends.base.generator import BaseKernelGenerator
from pyfr.backends.openmp.simd import NotVectorisable, SIMDLowering
from pyfr.dsl.codegen import CodeGenerator
from pyfr.dsl.nodes import Binary, DSLVar, Index, Int, Program, Var, VarDecl


class OpenMPCodeGenerator(CodeGenerator):
    # Vectorise the lane loops across the items of a SoA group
    region_markers = {'simd': '#pragma omp simd',
                      'unroll': '#pragma GCC unroll {0}'}


class OpenMPKernelGenerator(BaseKernelGenerator):
    codegen_cls = OpenMPCodeGenerator

    # Math functions we lower to helper functions in all kernels
    lower_fns = BaseKernelGenerator.lower_fns | {'exp', 'log'}

    # Lowerings for the @-intrinsics produced by the dereference rules
    _xidx = '_xi + _xj'
    _aosoa = '(_xi / @soasz*@nv + @v)*@soasz + _xj'
    _bcast = '@c % @ld + (@c / @ld)*@ld*@r'
    _tilein = ('(@r / @tr)*(@pc / @tc)*@tr*@tc'
               ' + (@c / @tc)*@tr*@tc + (@r % @tr)*@tc + @c % @tc')
    _tile = f'((_ib*BLK_SZ + _xi) / @soasz*@pr*@pc + {_tilein})*@soasz + _xj'
    _tileoff = f'({_tilein})*@soasz'

    # Prefix for variables in shared storage, which lives on the stack
    _shared_prfx = None

    # Largest lane-group size
    max_lanes = 1

    # Whether the body is vectorised across the items of a SoA group
    vectorised = False

    def _lane_decls(self):
        decls = self.lanemodel.decls

        # Give each item of the group its own copy of the storage
        sfx = f'[{self.soasz}]' if self.vectorised else ''

        return '\n'.join(f'{d.dtype} {d.name}{d.cdimstr}{sfx};' for d in decls)

    def _render_body_preamble_epilogue(self):
        self._staged_reduces = []
        self._bcol_reduces = []
        self._bcast_reduces = []
        return super()._render_body_preamble_epilogue()

    def _finalise_ast(self, ast):
        # Redirect 1D view reductions through their staging arrays
        for va in self.vectargs:
            if va.isreduce and va.isview:
                self._staged_reduces.append(va)
                ast = self._stage_reduce(ast, va)

        # Vectorise a cooperative body across the items of a SoA group
        if self.lanemodel.coop and not self.lanemodel.fenced:
            ast = self._lower_intrinsics(ast)
            shared = [d.name for d in self.lanemodel.decls]
            try:
                ast = SIMDLowering(self.soasz, shared).lower(ast)
                self.vectorised = True
            except NotVectorisable:
                pass

        return ast

    def _stage_reduce(self, ast, va):
        rv, xidx = Var(f'_rv_{va.name}'), DSLVar('xidx')

        def stage(node):
            node = node.map(stage)
            match node:
                # Rewrite name into _rv_name[@xidx]
                case Var(name) if name == va.name and not va.ismulti:
                    return Index(rv, xidx)
                # Rewrite name[i] into _rv_name[i][@xidx]
                case Index(Var(name), ix) if name == va.name:
                    return Index(Index(rv, ix), xidx)
                case _:
                    return node

        return stage(ast)

    def _render_reduce(self, va, body, codegen):
        decl, locs = self._reduce_locals(va.name, prod(va.cdims), va.ncdim)
        op, ra = va.reduceop, Var(f'_ra_{va.name}')

        # 2D broadcast-col: per-iteration local + direct accumulation
        if va.isbroadcastc:
            self._bcol_reduces.append(va)
            accum = [self._accum_stmt(op, self._reduce_target(va, j), l)
                     for j, l in enumerate(locs)]
        # Range reduction: per-item local + accumulator per lane of the block
        elif va.isbroadcast:
            self._bcast_reduces.append(va)
            accum = [self._accum_stmt(op, Index(Index(ra, Int(j)), Var('_xj')),
                                      l)
                     for j, l in enumerate(locs)]
        # 1D view reductions have already been staged by _finalise_ast
        else:
            return body

        dstr = self._generate(codegen, decl)
        astr = self._generate(codegen, Program(accum))
        return f'{dstr}\n{body}\n{astr}'

    def _render_bcast_reduce(self):
        # Declare and flush the block's accumulators of each range reduction
        codegen, soasz = self.codegen_cls(), self.soasz
        decls, flush = [], []

        for va in self._bcast_reduces:
            n = prod(va.cdims)
            ident, rv = self._reduce_ident(va.reduceop), f'_ra_{va.name}'
            decls.append(f'fpdtype_t {rv}[{n}][{soasz}];')
            decls.append(f'for (int _i = 0; _i < {n*soasz}; _i++) '
                         f'(&{rv}[0][0])[_i] = {ident};')

            # Combine the lanes of the block and pass the result on
            for j in range(n):
                src = Index(Index(Var(rv), Int(j)), Var('_xj'))
                accum = self._accum_stmt(va.reduceop, Var('_rv'), src)
                dst = Index(Var(f'{va.name}_v'), Int(j))
                atom = self._atomic_stmt(va.reduceop, dst, Var('_rv'))
                flush.append(f'''
                    {{
                        fpdtype_t _rv = {ident};
                        for (int _xj = 0; _xj < {soasz}; _xj++)
                            {self._generate(codegen, accum)}
                        {self._generate(codegen, atom)}
                    }}''')

        return '\n'.join(decls), '\n'.join(flush)

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
                decls.append(VarDecl('fpdtype_t', rv.name, sizes))
                for i in range(vs):
                    # Flush _rv_name[i][ix] into name_v[name_vix[vs*ix + i]]
                    vidx = Binary('+', Binary('*', Int(vs), ix), Int(i))
                    dst = Index(gv, Index(vix, vidx))
                    src = Index(Index(rv, Int(i)), ix)
                    atoms.append(self._atomic_stmt(va.reduceop, dst, src))
            else:
                # Declare the staging array fpdtype_t _rv_name[BLK_SZ]
                sizes = [Var('BLK_SZ')]
                decls.append(VarDecl('fpdtype_t', rv.name, sizes))
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
        kdecls, kflush = self._render_bcast_reduce()
        soasz = self.soasz

        # View reduce: atomic flush after each SIMD chunk
        if rflush:
            rflush = f'for (int _xj = 0; _xj < _xjn; _xj++) {{ {rflush} }}'

        # Broadcast-col reduce identity init
        rinit = self._render_bcol_reduce_init()

        # Give each item its own copy of the storage shared by its lanes
        body = f'{self._lane_decls()}\n{self.body}'
        simd = '' if self.lanemodel.coop else '#pragma omp simd'

        # Walk the lanes of a SoA group unless the body does so itself
        if not self.vectorised:
            body = f'''
                {simd}
                for (int _xj = 0; _xj < _xjn; _xj++)
                {{
                    {body}
                }}
                {rflush}'''

        def groups(bound, nxj):
            # Walk the SoA groups of a block, row by row for a 2D kernel
            loop = f'''
                for (int _xi = 0; _xi < {bound}; _xi += {soasz})
                {{
                    const int _xjn = {nxj};
                    {body}
                }}'''
            if self.ndim == 2:
                return f'''
                    for (ixdtype_t _y = 0; _y < _ny; _y++)
                    {{
                        {loop}
                    }}'''
            else:
                return loop

        # Walk every group of the full blocks and the rest of the partial block
        core = f'{rinit}\n{groups("BLK_SZ", soasz)}'
        clean = f'''
            {rinit}
            int _rem = _nx % BLK_SZ;
            {groups('_rem', f'min({soasz}, _rem - _xi)')}'''

        result = f'''
            {self.helpers}
            struct {self.name}_kargs {{ {kargdefn}; }};
            void {self.name}(ixdtype_t _ib,
                             const struct {self.name}_kargs *args,
                             int64_t _disp_mask)
            {{
                {kargassn};
                {rdecls}
                {kdecls}
                if (_nx - _ib*BLK_SZ >= BLK_SZ)
                {{
                    {core}
                }}
                else
                {{
                    {clean}
                }}
                {kflush}
            }}'''

        return result

    def ldim_size(self, name, factor=1):
        if factor > 1:
            return Binary('*', Int(factor), Var('BLK_SZ'))
        else:
            return Var('BLK_SZ')

    def needs_ldim(self, arg):
        return False

    def needs_blocksz(self, arg):
        if arg.isview or arg.istiled or arg.ismpi:
            return False
        # 2D broadcast vector
        elif self.ndim == 1:
            return not arg.isbroadcast
        # 2D broadcast vector or row broadcast matrix
        else:
            return not (arg.isbroadcast or arg.isbroadcastr)

    def _displace_arg(self, arg):
        # Step a matrix by the size of the blocks it is bound with
        if self.needs_blocksz(arg):
            return f'_ib*{arg.name}_blocksz'
        # Vector
        elif arg.ismpi:
            return '_ib*BLK_SZ'
        else:
            return None

    def _render_args(self, argn):
        # We first need the argument list; starting with the dimensions
        kargs = [('ixdtype_t', d, None, None) for d in self._dims]

        # Now add any scalar arguments
        kargs.extend((sa.dtype, sa.name, None, None) for sa in self.scalargs)

        # The displacement mask carries one bit per array argument
        if len(self.vectargs) > 63:
            raise ValueError('Kernels are limited to 63 array arguments')

        # Finally, add the vector arguments
        for vidx, va in enumerate(self.vectargs):
            da = self._displace_arg(va)
            mi = vidx if da else None

            if va.intent == 'in':
                kargs.append((f'const {va.dtype}*', f'{va.name}_v', da, mi))
            else:
                kargs.append((f'{va.dtype}*', f'{va.name}_v', da, mi))

            # Size of the blocks of a matrix
            if self.needs_blocksz(va):
                kargs.append(('ixdtype_t', f'{va.name}_blocksz', None, None))

            # Views
            if va.isview:
                vix_disp = f'_ib*BLK_SZ*{va.viewstride}'
                kargs.append(('const ixdtype_t*', f'{va.name}_vix', vix_disp,
                              None))

                if self.ndim == 2 and not va.isbroadcastc:
                    kargs.append(('ixdtype_t', f'{va.name}_vrstri', None,
                                  None))
                elif va.ncdim == 2 and va.cdims[0] > 1 and not va.istiled:
                    kargs.append(('const ixdtype_t*', f'{va.name}_vrstri',
                                  '_ib*BLK_SZ', None))

        # Argument definitions and assignments
        kargdefn, kargassn, kargdisp = [], [], []
        for dtype, name, disp, midx in kargs:
            assn = f'{dtype} {name} = {argn}->{name}'

            # Handle displacement and potential masking thereof
            if disp and midx is not None:
                assn += f' + ((_disp_mask & {1 << midx}) ? 0 : {disp})'
            elif disp:
                assn += f' + {disp}'

            kargdefn.append(f'{dtype} {name}')

            # Unpack the displaced pointers after the values they step by
            if disp:
                kargdisp.append(assn)
            else:
                kargassn.append(assn)

        return ';\n'.join(kargdefn), ';\n'.join(kargassn + kargdisp)
