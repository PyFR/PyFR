from math import prod

import numpy as np

from pyfr.backends.base.generator.args import Arg
from pyfr.backends.base.mathfns import lower_math_fns
from pyfr.dsl.ad import lower_ad_regions
from pyfr.dsl.codegen import CodeGenerator
from pyfr.dsl.nodes import (Assign, Binary, Call, DslCall, DslVar, ExprStmt,
                            Index, Int, Program, Unary, Var, VarDecl, map_ast,
                            region_kinds, walk_ast)
from pyfr.dsl.rewriter import (FloatSuffixer, Rewriter, parse_expr,
                               parse_program, rename_vars)
from pyfr.dsl.types import Environment


def _rule(tpl, **preconds):
    proto = parse_expr(tpl)
    dvars = {n.name for n in walk_ast(proto) if isinstance(n, DslVar)}
    nidx = sum(1 for v in dvars if v.startswith('I'))
    nvars = frozenset(v for v in dvars if v.startswith('N'))

    # Rules which reference the view stride are for multi-indexed views
    preconds['ismulti'] = 'VS' in dvars

    return proto, nidx, nvars, preconds


class BaseKernelGenerator:
    # Code generator class used to render the AST
    codegen_cls = CodeGenerator

    # Math functions we rewrite on all backends
    lower_fns = frozenset({'pow'})

    # Qualifier needed for helper functions to be callable from kernels
    _function_qual = ''

    # Lowerings for the @-intrinsics produced by the dereference rules
    _xidx = None
    _aosoa = None
    _bcast = None

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
        _rule('@N_v[@S*@I0 + @bcast(@V0, @I1, @S)]',
              isview=False, isbroadcast=True),
        # Stacked
        _rule('@N_v[@S*@I0 + @xidx]', isview=False, ismpi=False),
        # Doubly stacked
        _rule('@N_v[@S1*@I0 + @aosoa(@I1, @V1)]',
              isview=False, isbroadcast=False, ismpi=False),
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

    def __init__(self, backend, name, ndim, args, body):
        self.backend = backend
        self.name = name
        self.ndim = ndim
        self.fpdtype = backend.fpdtype
        self.ixdtype = backend.ixdtype
        self.soasz = backend.soasz
        self.fpdtype_max = float(np.finfo(self.fpdtype).max)

        # Parse and sort our argument list
        sargs = [Arg(k, v) for k, v in sorted(args.items())]

        # Argument data types may be typedefs defined outside the body
        atypes = {v.dtype for v in sargs}

        # Parse the body of our kernel into an AST
        self._ast = parse_program(body, types=atypes)

        # Differentiate the tangent and Jacobian regions of the body
        env = Environment()
        for v in sargs:
            env.define(v.name, v.dtype, [Int(d) for d in v.cdims] or None)

        self._ast = lower_ad_regions(self._ast, env)

        # Note the variables present in the body
        used = {n.name for n in walk_ast(self._ast) if isinstance(n, Var)}

        # Note the region kinds present in the body
        self.regions = region_kinds(self._ast)

        # Eliminate any arguments which go unreferenced in the body
        sargs = [v for v in sargs if v.name in used]

        # Break arguments into point-scalars and point-vectors
        self.scalargs = [v for v in sargs if v.isscalar]
        self.vectargs = vargs = [v for v in sargs if v.isvector]

        # Validate 2D argument constraints
        if ndim == 2:
            if any(v.isview and v.intent != 'in' for v in vargs):
                raise ValueError('2D view args must be input-only')
            if any(v.ismpi for v in vargs):
                raise ValueError('2D kernels do not support MPI matrices')

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
            elif self.needs_ldim(va):
                argt.append(('ml', [np.uintp, self.ixdtype]))
            else:
                argt.append(('m', [np.uintp]))

        # Return
        return self.ndim, argn, argt

    def ldim_size(self, name, factor=1):
        pass

    def needs_ldim(self, arg):
        pass

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

    def _render_reduce(self, va, body, codegen):
        n = prod(va.cdims) if va.cdims else 1

        # 2D broadcast-col: per-iteration local + register accumulator
        if va.isbroadcastc:
            rname = f'_ra_{va.name}'
            ident = self._reduce_ident(va.reduceop)

            if va.ncdim:
                pairs = [(Index(Var(rname), Int(j)), self._deref_arg(va, j))
                         for j in range(n)]

                rdecl = VarDecl(False, 'fpdtype_t', rname, [Int(n)])
                decl = VarDecl(False, 'fpdtype_t', va.name, [Int(n)])
                accum = [self._accum_stmt(va.reduceop, dst,
                                          Index(Var(va.name), Int(j)))
                         for j, (dst, _) in enumerate(pairs)]
            else:
                pairs = [(Var(rname), self._deref_arg(va))]

                rdecl = VarDecl(False, 'fpdtype_t', rname)
                decl = VarDecl(False, 'fpdtype_t', va.name)
                accum = [self._accum_stmt(va.reduceop, pairs[0][0],
                                          Var(va.name))]

            self._reduce_args.append((rdecl, pairs, ident, va.reduceop))

            dstr = self._generate(codegen, decl)
            astr = self._generate(codegen, Program(accum))
            return f'{dstr}\n{body}\n{astr}'
        # 1D: local variable + immediate atomic writeback
        else:
            vs = va.viewstride

            if vs > 1:
                decl = VarDecl(False, 'fpdtype_t', va.name, [Int(vs)])
                atoms = [self._atomic_stmt(va.reduceop, self._deref_arg(va, i),
                                           Index(Var(va.name), Int(i)))
                         for i in range(vs)]
            else:
                decl = VarDecl(False, 'fpdtype_t', va.name)
                atoms = [self._atomic_stmt(va.reduceop, self._deref_arg(va),
                                           Var(va.name))]

            dstr = self._generate(codegen, decl)
            astr = self._generate(codegen, Program(atoms))
            return f'{dstr}\n{body}\n{astr}'

    def _finalise_ast(self, ast):
        return ast

    def _lower_intrinsics(self, ast):
        xidx, soasz = parse_expr(self._xidx), Int(self.soasz)

        def lower(node):
            match node:
                case DslVar('xidx'):
                    return xidx
                case DslVar('soasz'):
                    return soasz
                case DslCall('aosoa', [v, nv]):
                    subs = {'v': lower(v), 'nv': lower(nv), 'soasz': soasz}
                    return rename_vars(parse_expr(self._aosoa), subs)
                case DslCall('bcast', [r, c, ld]):
                    subs = {'r': lower(r), 'c': lower(c), 'ld': lower(ld)}
                    return rename_vars(parse_expr(self._bcast), subs)
                case _:
                    return map_ast(node, lower)

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
        ast, helpers = lower_math_fns(ast, self.lower_fns)
        self._helper_calls |= helpers

        # At single precision suffix all floating point constants by 'f'
        if self.fpdtype == np.float32:
            ast = FloatSuffixer().transform(ast)

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
