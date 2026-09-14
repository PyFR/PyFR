from math import prod
import re

import numpy as np

from pyfr.cache import memoize
from pyfr.dsl.codegen import CodeGenerator
from pyfr.dsl.nodes import (Assign, Binary, Call, DslCall, DslVar, ExprStmt,
                            Float, Index, Int, Program, Unary, Var, VarDecl,
                            map_ast, region_kinds, unwrap_regions, walk_ast)
from pyfr.dsl.rewriter import (FloatSuffixer, Rewriter, fold_indices,
                               parse_expr, parse_program, rename_vars)
from pyfr.dsl.simplifier import Simplifier


def _rule(tpl, **preconds):
    proto = parse_expr(tpl)
    dvars = {n.name for n in walk_ast(proto) if isinstance(n, DslVar)}
    ncdim = sum(1 for v in dvars if v.startswith('I'))
    nvars = frozenset(v for v in dvars if v.startswith('N'))
    return proto, ncdim, nvars, preconds


class Arg:
    def __init__(self, name, spec):
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
    # Code generator class used to render the AST
    codegen_cls = CodeGenerator

    # Region kinds which are directives for the generator itself
    _unwrap_regions = frozenset({'simplify'})

    # Lowerings for the $-intrinsics produced by the dereference rules
    _xidx = None
    _aosoa = None
    _bcast = None

    _rules_1d = [
        # Views
        _rule('$N_v[$N_vix[$xidx]]', isview=True),
        _rule('$N_v[$N_vix[$xidx] + $soasz*$I0]', isview=True),
        _rule('$N_v[$N_vix[$xidx] + $N_vrstri[$xidx]*$I0 + $soasz*$I1]',
              isview=True),
        # Arrays
        _rule('$N_v[$xidx]', isview=False),
        # MPI
        _rule('$N_v[_nx*$I0 + $xidx]', isview=False, ismpi=True),
        _rule('$N_v[($V1*$I0 + $I1)*_nx + $xidx]', isview=False, ismpi=True),
        # Broadcast
        _rule('$N_v[$S*$I0 + $bcast($V0, $I1, $S)]',
              isview=False, isbroadcast=True),
        # Stacked
        _rule('$N_v[$S*$I0 + $xidx]', isview=False, ismpi=False),
        # Doubly stacked
        _rule('$N_v[$S1*$I0 + $aosoa($I1, $V1)]',
              isview=False, isbroadcast=False, ismpi=False),
    ]

    _rules_2d = [
        # Views
        _rule('$N_v[$N_vix[$xidx] + $N_vrstri*_y]', isview=True),
        _rule('$N_v[$N_vix[$xidx] + $N_vrstri*_y + $soasz*$I0]',
              isview=True),
        _rule('$N_v[$N_vix[$xidx] + $N_vrstri*($I0*_ny + _y) + $soasz*$I1]',
              isview=True),
        # Broadcast-col views (no _y dependence)
        _rule('$N_v[$N_vix[$xidx]]', isview=True, isbroadcastc=True),
        # Matrices
        _rule('$N_v[$xidx]', isview=False, isbroadcastc=True),
        _rule('$N_v[$S*_y + $xidx]', isview=False, isbroadcastc=False),
        # Broadcast
        _rule('$N_v[$S*$I0 + $bcast($V0, $I1, $S)]',
              isview=False, isbroadcast=True),
        # Broadcast-col
        _rule('$N_v[$S*$I0 + $xidx]', isview=False, isbroadcastc=True),
        # Broadcast-row
        _rule('$N_v[$S*_y + $bcast(_ny, $I0, $S)]',
              isview=False, isbroadcastr=True),
        # Stacked
        _rule('$N_v[$S0*_y + $aosoa($I0, $V0)]',
              isview=False, isbroadcastc=False, isbroadcastr=False),
        # Doubly stacked
        _rule('$N_v[$S1*$I0 + $aosoa($I1, $V1)]',
              isview=False, isbroadcastc=True),
        _rule('$N_v[($I0*_ny + _y)*$S1 + $aosoa($I1, $V1)]',
              isview=False, isbroadcast=False, isbroadcastc=False),
    ]

    def __init__(self, name, ndim, args, body, fpdtype, ixdtype, soasz):
        self.name = name
        self.ndim = ndim
        self.fpdtype = fpdtype
        self.ixdtype = ixdtype
        self.soasz = soasz
        self.fpdtype_max = float(np.finfo(fpdtype).max)

        # Parse and sort our argument list
        sargs = [Arg(k, v) for k, v in sorted(args.items())]

        # Argument data types may be typedefs defined outside the body
        atypes = {v.dtype for v in sargs}

        # Parse the body of our kernel into an AST
        self._ast = parse_program(body, types=atypes)

        # Note the variables present in the body
        used = {n.name for n in walk_ast(self._ast) if isinstance(n, Var)}

        # Region kinds are exposed for provider build-time decisions
        self.regions = region_kinds(self._ast)

        # Unwrap regions which are directives for the generator itself
        self._ast = unwrap_regions(self._ast, self._unwrap_regions)

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

        # Non-IEEE simplification is opted into by AD-generated kernels
        self._simplify = 'simplify' in self.regions

        # Select the dereference rules appropriate to our dimensionality
        if ndim == 1:
            self._deref_rules = self._deref_rules_1d
        else:
            self._deref_rules = self._deref_rules_2d

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
        proto, ncdim, nvars, preconds = compiled

        def apply(n, ix, arg):
            match = arg.ncdim == ncdim and len(ix) == ncdim
            if match and all(getattr(arg, a, None) == v
                             for a, v in preconds.items()):
                subs = {v: Var(n + v[1:]) for v in nvars}
                subs['S'] = self.ldim_size(n)
                for i, idx in enumerate(ix):
                    subs[f'I{i}'] = idx
                for i, cd in enumerate(arg.cdims):
                    subs[f'V{i}'] = Int(cd)
                    subs[f'S{i}'] = self.ldim_size(n, cd)

                return rename_vars(proto, subs)
            else:
                return None

        return apply

    def _deref_view_multi(self, n, ix, arg):
        # Multi-indexed views (viewstride > 1) bypass the rule tables
        if arg.viewstride > 1 and len(ix) == 1:
            vs = Int(arg.viewstride)
            vix = Binary('+', Binary('*', DslVar('xidx'), vs), ix[0])
            return Index(Var(f'{n}_v'), Index(Var(f'{n}_vix'), vix))
        else:
            return None

    @memoize
    def _deref_rules_1d(self):
        return [self._deref_view_multi,
                *(self._bind_rule(r) for r in self._rules_1d)]

    @memoize
    def _deref_rules_2d(self):
        return [self._deref_view_multi,
                *(self._bind_rule(r) for r in self._rules_2d)]

    def _deref_arg(self, arg, *ixs):
        # Convert the indices into AST nodes
        ix = [Int(i) if isinstance(i, int) else parse_expr(str(i))
              for i in ixs]

        for r in self._deref_rules():
            if (result := r(arg.name, ix, arg)) is not None:
                return fold_indices(result)

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

                decl = VarDecl(False, 'fpdtype_t', va.name, [Int(n)])
                accum = [self._accum_stmt(va.reduceop, dst,
                                          Index(Var(va.name), Int(j)))
                         for j, (dst, _) in enumerate(pairs)]
            else:
                pairs = [(Var(rname), self._deref_arg(va))]

                decl = VarDecl(False, 'fpdtype_t', va.name)
                accum = [self._accum_stmt(va.reduceop, pairs[0][0],
                                          Var(va.name))]

            self._reduce_args.append((rname, pairs, ident, va.reduceop))

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

    def _render_body(self, ast, codegen):
        # Fold constant arithmetic inside of array indices
        ast = fold_indices(ast)

        # Simplify the dereferenced expressions
        if self._simplify:
            ast = Simplifier().simplify_program(ast)

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
        ast = Rewriter(self._deref_rules()).transform(self._ast, args)

        return self._render_body(ast, self.codegen_cls()), '', ''


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

    # Prototype expressions for preloading 1D and 2D array arguments
    _preload_protos = {1: parse_expr('$S[$I0][$LX]'),
                       2: parse_expr('$S[$I0][$I1][$LX]')}

    # Lowerings for the $-intrinsics produced by the dereference rules
    _xidx = '_x'
    _aosoa = '(_x / $soasz*$nv + $v)*$soasz + _x % $soasz'
    _bcast = '$c'

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
        rules = preload_rules + self._deref_rules()
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
        for rn, pairs, ident, _ in self._reduce_args:
            iexpr = Float(float(ident))
            if (n := len(pairs)) > 1:
                stmts.append(VarDecl(False, 'fpdtype_t', rn, [Int(n)]))
                stmts += [ExprStmt(Assign(Index(Var(rn), Int(j)), iexpr))
                          for j in range(n)]
            else:
                stmts.append(VarDecl(False, 'fpdtype_t', rn, None, iexpr))
        preamble = self._generate(codegen, Program(stmts))

        # Epilogue: shared-memory reduce across y-threads, then
        # write result to global (single shared array, reused)
        rs = Index(Var('_rs'), Var(str(lx)))
        epilogue = f'{self._shared_prfx} fpdtype_t _rs[{bx}];'
        for rname, pairs, ident, reduceop in self._reduce_args:
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

        return f'''{spec}
            {{
                ixdtype_t _x = {self._gid};
                {self.preamble}
                {{
                    {self.body}
                }}
                {self.epilogue}
            }}'''
