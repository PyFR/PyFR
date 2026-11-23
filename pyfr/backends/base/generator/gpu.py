from pyfr.backends.base.generator.base import BaseKernelGenerator
from pyfr.backends.base.generator.lanes import StagedRowLanes
from pyfr.backends.base.generator.reduce import butterfly, gather
from pyfr.dsl.nodes import Assign, ExprStmt, Float, Program, Var
from pyfr.dsl.rewriter import Rewriter, rename_vars


class BaseGPUKernelGenerator(BaseKernelGenerator):
    # Block sizes for 1D and 2D kernels, respectively
    block1d = None
    block2d = None

    # Lane model class of the two dimensional kernels
    rowlanes_cls = StagedRowLanes

    # Expressions for local x/y id's and global x id
    _lid = None
    _gid = None

    # Prefix for variables in shared memory
    _shared_prfx = None

    # Declaration of a pointer argument, const-qualified where read-only
    _ptr_arg = '{c}{t}* {n}'

    # Expression for the index of a lane-group
    _grp = None

    # Largest lane-group size
    max_lanes = 256

    # Lowerings for the @-intrinsics produced by the dereference rules
    _xidx = '_x'
    _aosoa = '(_x / @soasz*@nv + @v)*@soasz + _x % @soasz'
    _bcast = '@c'
    _tile = ('_x*@pr*@pc + (@r / @tr)*(@pc / @tc)*@tr*@tc'
             ' + (@c / @tc)*@tr*@tc + (@r % @tr)*@tc + @c % @tc')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Prologue and loop of the launch shape
        self._prologue, loop = self._lane_indexing()

        # Specialise
        self.preamble += loop

    @property
    def meta(self):
        # Launch shape of the kernel
        launch = {'block': self.block, 'lanes': self.lanemodel.size}

        return super().meta | launch

    @property
    def block(self):
        # Count the lanes along each of the three axes of a block
        if self.lanemodel.coop:
            return (self.lanemodel.size, 1, 1)
        elif self.ndim == 1:
            return self.block1d
        else:
            return self.block2d

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
        if (arg.isbroadcast and arg.ncdim == 1) or arg.istiled:
            return False
        else:
            return self.ndim == 2 or (arg.ncdim > 0 and not arg.ismpi)

    def _render_body_preamble_epilogue(self):
        # Track 2D broadcast-col reductions for deferred writeback
        self._reduce_args = []

        codegen = self.codegen_cls()
        args = {va.name: va for va in self.vectargs if not va.isreduce}

        # Stage the column arguments of the block ahead of its rows
        prologue = Program(self.lanemodel.prologue)
        prologue = Rewriter(self._deref_rules).transform(prologue, args)
        preamble = self._generate(codegen, prologue)

        # Substitute the rules through the body
        ast = Rewriter(self._deref_rules).transform(self._ast, args)

        # Render the body
        body = self._render_body(ast, codegen)

        # Reductions: accumulator decls + shared-memory epilogue
        rpre, epilogue = self._render_reduce_epilogue(codegen)
        preamble += f'\n{rpre}\n'

        return body, preamble, epilogue

    def _render_reduce_epilogue(self, codegen):
        if not self._reduce_args:
            return '', ''

        # Preamble: register accumulators initialised to identity
        stmts = []
        for rdecl, pairs, ident, _ in self._reduce_args:
            iexpr = Float(float(ident))
            stmts.append(rdecl)
            stmts += [ExprStmt(Assign(acc, iexpr)) for acc, _ in pairs]
        preamble = self._generate(codegen, Program(stmts))

        # Epilogue: combine each accumulator across the lanes which share it
        bx, by = self.block[:2]
        epilogue = f'{self._shared_prfx} fpdtype_t _rs[{bx*by}];'
        for _, pairs, _, va in self._reduce_args:
            for acc, dst in pairs:
                epilogue += self._render_reduce_arg(codegen, va, acc, dst)

        return preamble, epilogue

    def _render_reduce_arg(self, codegen, va, acc, dst):
        bx, by = self.block[:2]
        lx, ly = self._lid
        lid = f'((int) {lx} + {bx}*(int) {ly})' if by > 1 else f'(int) {lx}'
        op, rv, prv = va.reduceop, Var('_rv'), Var('_p__rv')

        # Exchange across the block for a range reduction, a row for a column
        if va.isbroadcast:
            size, stride, xchg = bx*by, 1, butterfly
        else:
            size, stride, xchg = by, bx, gather

        # Fold the partner values into the accumulator at each exchange
        combine = [self._accum_stmt(op, rv, prv)]
        code = xchg(['_rv'], ['fpdtype_t'], ['_rs'], size, combine, stride)
        code = rename_vars(Program(code), {'lidx': Var(lid)})
        code = self._generate(codegen, code)

        # Have the first lane of the exchange pass the result on
        if va.isbroadcast:
            tail = self._generate(codegen, self._atomic_stmt(op, dst, rv))
            tail = f'if ({lid} == 0) {tail}'
        else:
            tail = self._generate(codegen, ExprStmt(Assign(dst, rv)))
            tail = f'if ({ly} == 0 && _x < _nx) {tail}'

        return f'''
            {{
                fpdtype_t _rv = {self._generate(codegen, acc)};
                {code}
                {tail}
            }}'''

    def _lane_decls(self):
        prfx, model = self._shared_prfx, self.lanemodel

        return '\n'.join(f'{prfx} {d.dtype} {d.name}{model.extents(d)};'
                         for d in model.decls)

    def _lane_indexing(self):
        # Have the lane-group of the block take an item of the range
        if self.lanemodel.coop:
            pro, loop = f'ixdtype_t _x = {self._grp};', ''
        # Have each lane take an item of the range
        elif self.ndim == 1:
            pro = f'ixdtype_t _x = {self._gid};'
            loop = 'if (_x < _nx)'
        # Have each lane take a share of the rows of an item of the range
        else:
            blk_y, lid_y = self.block2d[1], self._lid[1]
            pro = f'ixdtype_t _x = {self._gid};'
            loop = f'''
                ixdtype_t _ysize = (_ny + {blk_y - 1}) / {blk_y};
                ixdtype_t _ystart = {lid_y}*_ysize;
                ixdtype_t _yend = min(_ny, _ystart + _ysize);
                for (ixdtype_t _y = _ystart; _x < _nx && _y < _yend; _y++)'''

        return pro, loop

    def render(self):
        spec = self._render_spec()

        return f'''{self.helpers}
            {spec}
            {{
                {self._lane_decls()}
                {self._prologue}
                {self.preamble}
                {{
                    {self.body}
                }}
                {self.epilogue}
            }}'''
