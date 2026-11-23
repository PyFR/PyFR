from math import prod

import numpy as np

from pyfr.backends.base.generator.args import Arg
from pyfr.backends.base.generator.lanes import RowLanes, make_lanemodel
from pyfr.backends.base.mathfns import MathFnLowering
from pyfr.dsl.ad import ADLowering
from pyfr.dsl.codegen import CodeGenerator
from pyfr.dsl.nodes import (Assign, Binary, Call, DSLCall, DSLVar, ExprStmt,
                            Index, Int, Program, Region, Unary, Var, VarDecl,
                            walk_ast)
from pyfr.dsl.rewriter import (FloatSuffixer, Rewriter, parse_expr,
                               parse_program, rename_vars)
from pyfr.dsl.types import Environment
from pyfr.nputil import bfloat16


def _rule(tpl, **preconds):
    proto = parse_expr(tpl)
    dvars = {n.name for n in walk_ast(proto) if isinstance(n, DSLVar)}
    nidx = sum(1 for v in dvars if v.startswith('I'))
    nvars = frozenset(v for v in dvars if v.startswith('N'))

    # Rules which reference the view stride are for multi-indexed views
    preconds['ismulti'] = 'VS' in dvars

    return proto, nidx, nvars, preconds


class BaseKernelGenerator:
    # Code generator class used to render the AST
    codegen_cls = CodeGenerator

    # Lane model class of the two dimensional kernels
    rowlanes_cls = RowLanes

    # Math functions we rewrite on all backends
    lower_fns = frozenset({'pow'})

    # Qualifier needed for helper functions to be callable from kernels
    _function_qual = ''

    # NumPy data types of the backend independent DSL scalar types
    _npdtypes = {
        'int8_t': np.int8, 'uint8_t': np.uint8, 'int16_t': np.int16,
        'uint16_t': np.uint16, 'int': np.int32, 'int32_t': np.int32,
        'uint32_t': np.uint32, 'int64_t': np.int64, 'uint64_t': np.uint64,
        'ssize_t': np.intp, 'size_t': np.uintp, 'float': np.float32,
        'double': np.float64, 'half': np.float16, 'bf16': bfloat16
    }

    # Lowerings for the @-intrinsics produced by the dereference rules
    _xidx = None
    _aosoa = None
    _bcast = None
    _tile = None

    # Parameters of the @-intrinsics lowered through the string templates
    _intrinsic_params = {
        'aosoa': ('v', 'nv'),
        'bcast': ('r', 'c', 'ld'),
        'tile': ('r', 'c', 'tr', 'tc', 'pr', 'pc'),
    }

    _rules_1d = [
        # Views
        _rule('@N_v[@N_vix[@xidx]]', isview=True),
        _rule('@N_v[@N_vix[@xidx] + @soasz*@I0]', isview=True),
        _rule('@N_v[@N_vix[@xidx] + @N_vrstri[@xidx]*@I0 + @soasz*@I1]',
              isview=True),
        # Multi-indexed views (viewstride > 1)
        _rule('@N_v[@N_vix[@xidx*@VS + @I0]]', isview=True),
        # Arrays
        _rule('@N_v[@xidx]', isview=False),
        # MPI
        _rule('@N_v[_nx*@I0 + @xidx]', isview=False, ismpi=True),
        _rule('@N_v[(@V1*@I0 + @I1)*_nx + @xidx]', isview=False, ismpi=True),
        # Broadcast
        _rule('@N_v[@I0]', isview=False, isbroadcast=True),
        _rule('@N_v[@S*@I0 + @bcast(@V0, @I1, @S)]',
              isview=False, isbroadcast=True),
        # Tiled
        _rule('@N_v[@tile(@I0, @I1, @TR, @TC, @PR, @PC)]', istiled=True),
        # Stacked
        _rule('@N_v[@S*@I0 + @xidx]', isview=False, ismpi=False),
        # Doubly stacked
        _rule('@N_v[@S1*@I0 + @aosoa(@I1, @V1)]',
              isview=False, isbroadcast=False, ismpi=False, istiled=False),
    ]

    _rules_2d = [
        # Views
        _rule('@N_v[@N_vix[@xidx] + @N_vrstri*_y]', isview=True,
              isbroadcastc=False),
        _rule('@N_v[@N_vix[@xidx] + @N_vrstri*_y + @soasz*@I0]',
              isview=True, isbroadcastc=False),
        _rule('@N_v[@N_vix[@xidx] + @N_vrstri*(@I0*_ny + _y) + @soasz*@I1]',
              isview=True, isbroadcastc=False),
        # Broadcast-col views (no _y dependence)
        _rule('@N_v[@N_vix[@xidx]]', isview=True, isbroadcastc=True),
        _rule('@N_v[@N_vix[@xidx*@VS + @I0]]', isview=True,
              isbroadcastc=True),
        # Matrices
        _rule('@N_v[@xidx]', isview=False, isbroadcastc=True),
        _rule('@N_v[@S*_y + @xidx]', isview=False, isbroadcastc=False),
        # Broadcast
        _rule('@N_v[@I0]', isview=False, isbroadcast=True),
        _rule('@N_v[@S*@I0 + @bcast(@V0, @I1, @S)]',
              isview=False, isbroadcast=True),
        # Broadcast-col
        _rule('@N_v[@S*@I0 + @xidx]', isview=False, isbroadcastc=True),
        # Broadcast-row
        _rule('@N_v[@S*_y + @bcast(_ny, @I0, @S)]',
              isview=False, isbroadcastr=True),
        # Stacked
        _rule('@N_v[@S0*_y + @aosoa(@I0, @V0)]',
              isview=False, isbroadcastc=False, isbroadcastr=False),
        # Doubly stacked
        _rule('@N_v[@S1*@I0 + @aosoa(@I1, @V1)]',
              isview=False, isbroadcastc=True),
        _rule('@N_v[(@I0*_ny + _y)*@S1 + @aosoa(@I1, @V1)]',
              isview=False, isbroadcast=False, isbroadcastc=False),
    ]

    def __init__(self, backend, name, ndim, args, body, coop=None):
        self.backend = backend
        self.name = name
        self.ndim = ndim
        self.fpdtype = backend.fpdtype
        self.ixdtype = backend.ixdtype
        self.soasz = backend.soasz
        self.fpdtype_max = float(np.finfo(self.fpdtype).max)

        # NumPy data types of the DSL scalar types
        self._npdtypes = self._npdtypes | {
            'fpdtype_t': self.fpdtype, 'ixdtype_t': self.ixdtype
        }

        # Parse and sort our argument list
        sargs = [Arg(k, v) for k, v in sorted(args.items())]

        # Lanes assigned to each item of the iteration range
        self.lanemodel = make_lanemodel(self, coop)

        # Validate the arguments against our dimensionality and lanes
        self._validate_args([v for v in sargs if v.isvector])

        # Parse and lower the body of our kernel into an AST
        self._ast = self._parse_body(body, sargs)

        # Note the variables present in the body
        nodes = list(walk_ast(self._ast))
        used = {n.name for n in nodes if isinstance(n, Var)}

        # Keep the arguments staged by the block, which the body may not name
        used |= self.lanemodel.staged

        # Note the region kinds present in the body
        kinds = {n.kind for n in nodes if isinstance(n, Region)}
        self.regions = frozenset(kinds)

        # Eliminate any arguments which go unreferenced in the body
        sargs = [v for v in sargs if v.name in used]

        # Break arguments into point-scalars and point-vectors
        self.scalargs = [v for v in sargs if v.isscalar]
        self.vectargs = vargs = [v for v in sargs if v.isvector]

        # Math helper functions required by the kernel
        self._helper_calls = {}

        # Select the dereference rules appropriate to our dimensionality
        rules = self._rules_1d if ndim == 1 else self._rules_2d
        self._deref_rules = [self._bind_rule(r) for r in rules]

        # Render the main body of our kernel
        body, preamble, epilogue = self._render_body_preamble_epilogue()
        self.body, self.preamble, self.epilogue = body, preamble, epilogue

        # Determine the dimensions to be iterated over
        self._dims = ['_nx'] if ndim == 1 else ['_ny', '_nx']

    def _parse_body(self, body, sargs):
        # Argument data types may be typedefs defined outside the body
        atypes = {v.dtype for v in sargs}

        # Parse the body of our kernel into an AST
        ast = parse_program(body, types=atypes)

        # Define the types of the arguments as seen by the body
        env = Environment()
        for v in sargs:
            env.define(v.name, v.dtype, [Int(d) for d in v.cdims] or None)

        # Differentiate the tangent and Jacobian regions of the body
        ast = ADLowering(env).visit(ast)

        # Hoist the shared storage declarations out of the body
        args = {v.name: v for v in sargs}
        ast = self.lanemodel.hoist_shared(ast, args)

        # Lower the lane intrinsics, barriers and reductions of the body
        return self.lanemodel.lower(ast, env)

    def _validate_args(self, vargs):
        # Validate 2D argument constraints
        if self.ndim == 2:
            if any(v.isview and v.intent != 'in' for v in vargs):
                raise ValueError('2D view args must be input-only')
            if any(v.ismpi for v in vargs):
                raise ValueError('2D kernels do not support MPI matrices')
            if any(v.istiled for v in vargs):
                raise ValueError('2D kernels do not support tiled matrices')

        # Validate the argument constraints of our lanes
        self.lanemodel.validate_args(vargs)

    def argspec(self):
        # Argument names plus their form and types
        argn, argt = [], []

        # Dimensions
        argn += self._dims
        argt += [('s', [self.ixdtype])]*self.ndim

        # Scalar args (fpdtype or ixdtype)
        for sa in self.scalargs:
            argn.append(sa.name)
            dtype = self.ixdtype if sa.dtype == 'ixdtype_t' else self.fpdtype
            argt.append(('s', [dtype]))

        # Vector args
        for va in self.vectargs:
            argn.append(va.name)

            if va.isview:
                match self.ndim, va.ncdim:
                    case 2, _ if va.isbroadcastc:
                        argt.append(('v', [np.uintp]*2))
                    case 2, _:
                        argt.append(('vs', [np.uintp]*2 + [self.ixdtype]))
                    case _, 2 if va.cdims[0] > 1:
                        argt.append(('va', [np.uintp]*3))
                    case _:
                        argt.append(('v', [np.uintp]*2))
            elif va.istiled:
                argt.append(('t', [np.uintp]))
            elif self.needs_ldim(va):
                argt.append(('ml', [np.uintp, self.ixdtype]))
            elif self.needs_blocksz(va):
                argt.append(('mb', [np.uintp, self.ixdtype]))
            else:
                argt.append(('m', [np.uintp]))

        # Return
        return self.ndim, argn, argt

    @property
    def meta(self):
        # Metadata with which the provider builds and launches the kernel
        return {'regions': self.regions}

    def tiling(self, arg):
        # Find the tile shape of the argument and its extents in whole tiles
        tr, tc = self.backend.optimal_tile_shape(self.npdtype(arg.dtype))
        padr, padc = -(-arg.cdims[0] // tr)*tr, -(-arg.cdims[1] // tc)*tc

        return tr, tc, padr, padc

    def npdtype(self, ctype):
        return np.dtype(self._npdtypes[ctype])

    def ldim_size(self, name, factor=1):
        pass

    def needs_ldim(self, arg):
        pass

    def needs_blocksz(self, arg):
        return False

    def render(self):
        pass

    def _bind_rule(self, compiled):
        proto, nidx, nvars, preconds = compiled

        def apply(n, ix, arg):
            flags = all(getattr(arg, a) == v for a, v in preconds.items())
            if len(ix) == nidx == arg.nidx and flags:
                subs = {v: Var(n + v[1:]) for v in nvars}
                subs['S'] = self.ldim_size(n)
                subs['VS'] = Int(arg.viewstride)
                for i, idx in enumerate(ix):
                    subs[f'I{i}'] = idx
                for i, cd in enumerate(arg.cdims):
                    subs[f'V{i}'] = Int(cd)
                    subs[f'S{i}'] = self.ldim_size(n, cd)

                # Bind the tile shape and padded extents of a tiled argument
                if arg.istiled:
                    tiling = zip(('TR', 'TC', 'PR', 'PC'), self.tiling(arg))
                    subs |= {k: Int(v) for k, v in tiling}

                return rename_vars(proto, subs)
            else:
                return None

        return apply

    def _deref_arg(self, arg, *ixs):
        # Convert the indices into AST nodes
        ix = [Int(i) if isinstance(i, int) else parse_expr(str(i))
              for i in ixs]

        for r in self._deref_rules:
            if (result := r(arg.name, ix, arg)) is not None:
                return result

        raise ValueError(f'No matching rule for {arg.name}')

    def _reduce_ident(self, reduceop):
        m = self.fpdtype_max
        return {'sum': 0, 'min': m, 'max': -m}[reduceop]

    _reduce_fn = {'min': 'fmin', 'max': 'fmax'}

    def _accum_stmt(self, reduceop, dst, src):
        if reduceop == 'sum':
            return ExprStmt(Assign(dst, Binary('+', dst, src)))
        else:
            rfn = Var(self._reduce_fn[reduceop])
            return ExprStmt(Assign(dst, Call(rfn, [dst, src])))

    def _atomic_stmt(self, reduceop, dst, src):
        fn = Var(f'atomic_{reduceop}_fpdtype')
        return ExprStmt(Call(fn, [Unary('&', dst), src]))

    def _reduce_target(self, va, j):
        # Find the entry of the output an accumulated value is combined into
        if va.ncdim:
            return self._deref_arg(va, j)
        else:
            return self._deref_arg(va)

    def _reduce_locals(self, name, n, isarray):
        # Declare the local accumulator of a reduction and list its entries
        if isarray:
            decl = VarDecl('fpdtype_t', name, [Int(n)])
            return decl, [Index(Var(name), Int(j)) for j in range(n)]
        else:
            return VarDecl('fpdtype_t', name), [Var(name)]

    def _render_reduce(self, va, body, codegen):
        op, n = va.reduceop, prod(va.cdims)

        # Column and range reductions: per-item local + register accumulator
        if va.isbroadcastc or va.isbroadcast:
            rdecl, regs = self._reduce_locals(f'_ra_{va.name}', n, va.ncdim)
            pairs = [(r, self._reduce_target(va, j))
                     for j, r in enumerate(regs)]
            self._reduce_args.append((rdecl, pairs, self._reduce_ident(op),
                                      va))

            decl, locs = self._reduce_locals(va.name, n, va.ncdim)
            accum = [self._accum_stmt(op, r, l) for r, l in zip(regs, locs)]
        # Views: local variable + immediate atomic writeback
        else:
            vs = va.viewstride
            decl, locs = self._reduce_locals(va.name, vs, vs > 1)
            if vs > 1:
                dsts = [self._deref_arg(va, i) for i in range(vs)]
            else:
                dsts = [self._deref_arg(va)]
            accum = [self._atomic_stmt(op, d, l) for d, l in zip(dsts, locs)]

        dstr = self._generate(codegen, decl)
        astr = self._generate(codegen, Program(accum))
        return f'{dstr}\n{body}\n{astr}'

    def _finalise_ast(self, ast):
        return ast

    def _lower_intrinsics(self, ast):
        xidx, soasz = parse_expr(self._xidx), Int(self.soasz)

        def lower(node):
            match node:
                case DSLVar('xidx'):
                    return xidx
                case DSLVar('soasz'):
                    return soasz
                case DSLCall(fn, args) if fn in self._intrinsic_params:
                    tpl = parse_expr(getattr(self, f'_{fn}'))
                    params = self._intrinsic_params[fn]
                    subs = dict(zip(params, map(lower, args)), soasz=soasz)
                    return rename_vars(tpl, subs)
                case _:
                    return node.map(lower)

        return lower(ast)

    def _generate(self, codegen, ast):
        return codegen.generate(self._lower_intrinsics(ast))

    @property
    def helpers(self):
        # Generate any math helper functions used by the body
        return '\n'.join(h.render(self.fpdtype, self._function_qual)
                         for h in self._helper_calls.values())

    def _render_body(self, ast, codegen):
        # Lower math functions to helper function calls
        mfl = MathFnLowering(self.lower_fns)
        ast = mfl.visit(ast)
        self._helper_calls |= mfl.helpers

        # At single precision suffix all floating point constants by 'f'
        if self.fpdtype == np.float32:
            ast = FloatSuffixer().visit(ast)

        # Apply any backend-specific finalisation passes
        ast = self._finalise_ast(ast)

        # Generate the code for the body
        body = self._generate(codegen, ast)

        # Handle any reduction arguments
        for va in self.vectargs:
            if va.isreduce:
                body = self._render_reduce(va, body, codegen)

        return body

    def _render_body_preamble_epilogue(self):
        # Track 2D broadcast-col reductions for deferred writeback
        self._reduce_args = []

        # Substitute the dereference rules through the body
        args = {va.name: va for va in self.vectargs if not va.isreduce}
        ast = Rewriter(self._deref_rules).transform(self._ast, args)

        return self._render_body(ast, self.codegen_cls()), '', ''
