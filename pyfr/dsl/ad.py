from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product

from pyfr.dsl.codegen import CodeGenerator
from pyfr.dsl.derivs import Differentiator, build_index, extract_loop_offset
from pyfr.dsl.nodes import (ArrayInit, Assign, Block, DoWhile, ExprStmt, For,
                            If, Index, Int, MultiDecl, Number, Postfix,
                            Program, Region, Unary, Var, VarDecl, While,
                            expr_children, map_ast, unwrap_index, walk_ast)
from pyfr.dsl.rewriter import rename_vars
from pyfr.dsl.simplifier import Simplifier
from pyfr.dsl.types import DOUBLE, Environment


def _collect_array_accesses(expr, array_name):
    match expr:
        case Index(Var(name), index) if name == array_name:
            return [index]
        case _:
            return [i for c in expr_children(expr)
                    for i in _collect_array_accesses(c, array_name)]


def _target_name(stmt):
    match stmt:
        case VarDecl(_, _, name, _, _):
            return name
        case ExprStmt(Assign(lhs, _)):
            name, _ = unwrap_index(lhs)
            return name
        case _:
            return None


def _is_zero(expr):
    match expr:
        case Number(v):
            return v == 0
        case _:
            return False


def _int_key(idxs):
    if all(isinstance(ix, Int) for ix in idxs):
        return tuple(ix.value for ix in idxs)
    else:
        return None


def _check_pure(expr):
    # Reject expressions whose side effects the tile transform cannot see
    match expr:
        case Assign() | Postfix() | Unary('++' | '--', _):
            raise NotImplementedError('Side effects are not supported in '
                                      'tile generation')
        case _:
            for c in expr_children(expr):
                _check_pure(c)


class DerivativeGenerator:
    # Region kinds whose rewrite guarantees differentiation would break
    nodiff_regions = frozenset({'fp-precise'})

    def __init__(self, wrt, prefix='d_', tangent_mode=False, array_info={},
                 force_arrays=set()):
        self.wrt = {wrt} if isinstance(wrt, str) else set(wrt)
        self.prefix = prefix
        self.tangent_mode = tangent_mode
        self.array_info = array_info
        self.force_arrays = force_arrays
        self.dependent_vars = defaultdict(set)
        self.dependent_arrays = defaultdict(set)
        self.simplifier = Simplifier(Environment(default=DOUBLE))
        self.codegen = CodeGenerator()
        self.loop_var = None

    def _analyse(self, program):
        # Gather the target and value of every write in the program
        writes = []
        for node in walk_ast(program):
            match node:
                case VarDecl(_, vtype, name, sizes, init):
                    # Let the simplifier see the declared type of each name
                    self.simplifier.env.define(name, vtype)
                    if init is not None:
                        writes.append((name, sizes is not None, init))
                case Assign(lhs, rhs):
                    name, indices = unwrap_index(lhs)
                    if name is not None:
                        writes.append((name, bool(indices), rhs))

        # Propagate the dependence on each input through the writes
        self.dependent_vars = defaultdict(set)
        self.dependent_arrays = defaultdict(set)
        for w in self.wrt:
            deps, darrs = self.dependent_vars[w], self.dependent_arrays[w]
            diff = self._dependence_checker(w)
            indirect = Differentiator(None, deps, dependent_arrays=darrs)
            rowwise = not self.tangent_mode and w in self.array_info

            ndeps = -1
            while ndeps != len(deps) + len(darrs):
                ndeps = len(deps) + len(darrs)
                for name, is_array, rhs in writes:
                    if diff.contains_var(rhs):
                        # Reject Jacobian rows not seeded by direct accesses
                        dense = not is_array or indirect.contains_var(rhs)
                        if rowwise and dense:
                            raise NotImplementedError(
                                f'{name!r} depends on {w!r} through other '
                                'variables; use tangent mode'
                            )
                        elif is_array:
                            darrs.add(name)
                        else:
                            deps.add(name)

    def _dependence_checker(self, w):
        # Build the checker for dependence on the input w
        avars = self.wrt if self.tangent_mode else set(self.array_info)
        return Differentiator(w, self.dependent_vars[w], array_vars=avars,
                              dependent_arrays=self.dependent_arrays[w],
                              tangent_mode=self.tangent_mode)

    def _make_differentiator(self, input_var=None, seed_index=None):
        # Tangent mode differentiates all inputs simultaneously
        if self.tangent_mode:
            deps = set().union(*self.dependent_vars.values())
            darrs = set().union(*self.dependent_arrays.values())

            def namer(n): return f'{self.prefix}{n}'

            return Differentiator(None, deps, namer, array_vars=self.wrt,
                                  dependent_arrays=darrs, tangent_mode=True)
        # Otherwise differentiate with respect to the given input alone
        else:
            deps = self.dependent_vars[input_var]
            darrs = self.dependent_arrays[input_var]
            if len(self.wrt) == 1:
                def namer(n): return f'{self.prefix}{n}'
            else:
                def namer(n): return f'{self.prefix}{n}_{input_var}'

            return Differentiator(input_var, deps, namer,
                                  array_vars=set(self.array_info),
                                  dependent_arrays=darrs, tangent_mode=False)

    def _depends_on_any(self, expr):
        return any(self._dependence_checker(w).contains_var(expr)
                   for w in self.wrt)

    def _wrts(self, name):
        # Find the inputs the variable depends on, merged in tangent mode
        ws = [w for w in sorted(self.wrt)
              if name in self.dependent_vars[w] | self.dependent_arrays[w]]
        if self.tangent_mode:
            return [None] if ws else []
        else:
            return ws

    def _deriv_name(self, name, w=None):
        if self.tangent_mode or len(self.wrt) == 1:
            return f'{self.prefix}{name}'
        else:
            return f'{self.prefix}{name}_{w}'

    def _diff_and_simplify(self, expr, input_var=None):
        diff = self._make_differentiator(input_var)

        return self.simplifier.simplify_fully(diff.diff(expr))

    def _transform_stmt(self, stmt):
        match stmt:
            # Multi-declarations are flattened and handled individually
            case MultiDecl(decls):
                return [s for d in decls for s in self._transform_stmt(d)]
            # Variable and array declarations
            case VarDecl(_, vtype, name, sizes, init):
                return self._transform_decl(stmt, vtype, name, sizes, init)
            # Scalar variable assignments
            case ExprStmt(Assign(Var(name), expr)):
                return self._transform_assign(stmt, name, expr)
            # Array element assignments
            case ExprStmt(Assign(Index() as lhs, expr)):
                return self._transform_array_assign(stmt, lhs, expr)
            # Block statements
            case Block(body):
                stmts = [s for st in body for s in self._transform_stmt(st)]
                return [Block(stmts)]
            # Conditional statements
            case If(cond, then, else_):
                def tf(s): return self._wrap_block(self._transform_stmt(s))

                return [If(cond, tf(then), tf(else_) if else_ else None)]
            # Loop statements
            case While(cond, body):
                tbody = self._wrap_block(self._transform_stmt(body))
                return [While(cond, tbody)]
            case For(init, cond, step, body):
                oldlv = self.loop_var

                # Track loop variable for array access analysis
                match init:
                    case VarDecl(_, _, name, _, _):
                        self.loop_var = name
                    case ExprStmt(Assign(Var(name), _)):
                        self.loop_var = name

                tbody = self._wrap_block(self._transform_stmt(body))
                self.loop_var = oldlv
                return [For(init, cond, step, tbody)]
            case DoWhile(body, cond):
                tbody = self._wrap_block(self._transform_stmt(body))
                return [DoWhile(tbody, cond)]
            # Differentiate region bodies, preserving the wrapper
            case Region(kind, body, args):
                if kind in self.nodiff_regions:
                    raise ValueError(f'Cannot differentiate a {kind} region')

                stmts = [s for st in body for s in self._transform_stmt(st)]
                return [Region(kind, stmts, args)]
            # Pass through other statements unchanged
            case _:
                return [stmt]

    def _transform_decl(self, stmt, vtype, name, sizes, init):
        # Array declarations with initialisers must be lowered first
        if sizes and init and self._depends_on_any(init):
            raise NotImplementedError('Cannot differentiate initialised '
                                      f'array {name!r}')

        out = [stmt]
        for w in self._wrts(name):
            # Initialise the tangent from the initialiser, else to zero
            if sizes:
                dinit = ArrayInit([Int(0)])
            elif init:
                dinit = self._diff_and_simplify(init, w)
            else:
                dinit = Int(0)

            dn = self._deriv_name(name, w)
            out.append(VarDecl(False, vtype, dn, sizes, dinit))

        return out

    def _transform_assign(self, stmt, name, expr):
        tangents = []
        for w in self._wrts(name):
            deriv = self._diff_and_simplify(expr, w)
            dvar = Var(self._deriv_name(name, w))
            tangents.append(ExprStmt(Assign(dvar, deriv)))

        # Tangents precede the write so they read pre-update values
        return tangents + [stmt]

    def _transform_array_assign(self, stmt, lhs, expr):
        aname, indices = unwrap_index(lhs)

        # Force a tangent for writes to forced/input arrays
        force = self.tangent_mode and (aname in self.force_arrays or
                                       aname in self.array_info)
        ws = self._wrts(aname) or ([None] if force else [])

        tangents = []
        for w in ws:
            dname = self._deriv_name(aname, w)

            # Seed input array derivatives once per access
            if w in self.array_info:
                tangents += self._seed_derivs(expr, w, dname, indices)
            # Otherwise differentiate the expression directly
            else:
                deriv = self._diff_and_simplify(expr, w)
                dlhs = build_index(Var(dname), indices)
                tangents.append(ExprStmt(Assign(dlhs, deriv)))

        # Tangents precede the write so they read pre-update values
        return tangents + [stmt]

    def _seed_derivs(self, expr, w, dname, indices):
        out, seen = [], []
        deps, darrs = self.dependent_vars[w], self.dependent_arrays[w]
        avars = set(self.array_info)

        for aidx in _collect_array_accesses(expr, w):
            # Seed each distinct element once, keyed by offset or index
            off = extract_loop_offset(aidx, self.loop_var)
            key = aidx if off is None else off
            if key in seen:
                continue

            seen.append(key)

            def namer(n, w=w): return self._deriv_name(n, w)
            diff = Differentiator(w, deps, namer, array_vars=avars,
                                  dependent_arrays=darrs,
                                  loop_var=self.loop_var, seed_index=aidx)
            deriv = self.simplifier.simplify_fully(diff.diff(expr))
            if not _is_zero(deriv):
                dlhs = build_index(Var(dname), indices + [aidx])
                out.append(ExprStmt(Assign(dlhs, deriv)))

        return out

    def _wrap_block(self, stmts):
        if len(stmts) == 1:
            return stmts[0]
        else:
            return Block(stmts)

    def generate_derivative_program(self, program):
        self._analyse(program)
        stmts = [s for st in program.body for s in self._transform_stmt(st)]

        return Program(stmts)

    def generate_code(self, program):
        return self.codegen.generate(self.generate_derivative_program(program))


@dataclass(slots=True)
class ColumnState:
    # Live scalar and array element tangents for the column
    scalars: set = field(default_factory=set)
    arrays: set = field(default_factory=set)

    # Out tile elements written and seed elements overwritten
    outs: set = field(default_factory=set)
    written: set = field(default_factory=set)

    # Tangent declarations made for the column
    declared: set = field(default_factory=set)


class JacobianGenerator:
    def __init__(self, wrt, outs, prefix='d_', sizes={}):
        self.wrt = dict(wrt)
        self._shapes = {a: (n,) if isinstance(n, int) else tuple(n)
                        for a, n in self.wrt.items()}
        self.outs = set(outs)
        self.prefix = prefix

        # Shapes of extern arrays written by the program
        self.sizes = sizes
        self.simplifier = Simplifier(Environment(default=DOUBLE))
        self.codegen = CodeGenerator()

    def generate_tile_program(self, program):
        # Emits primal code plus exact Jacobian tile columns per input
        self._bidx = 0
        stmts = self._flatten_stmts(program.body)
        stmts = self._lower_array_inits(stmts)
        self._validate(stmts)

        gen = DerivativeGenerator(wrt=set(self.wrt), prefix=self.prefix,
                                  tangent_mode=True, array_info=dict(self.wrt),
                                  force_arrays=self.outs)
        dprog = gen.generate_derivative_program(Program(stmts))

        # Collect the names of all tangent variables from the transform
        deps = set(self.wrt).union(self.outs, *gen.dependent_vars.values(),
                                   *gen.dependent_arrays.values())
        arrdeps = set().union(*gen.dependent_arrays.values())
        self._tannames = {f'{self.prefix}{n}' for n in deps}
        self._tanarrs = {f'{self.prefix}{n}' for n in arrdeps}
        self._seed_arrs = {f'{self.prefix}{n}': n for n in self.wrt}
        self._outtiles = {f'{self.prefix}{n}': n for n in self.outs}

        # Record declarations for materialising tangent temporaries
        self._decls = decls = {n: ('fpdtype_t', [Int(s) for s in shape])
                               for n, shape in self._shapes.items()}
        decls |= {n: ('fpdtype_t', [Int(s) for s in sz])
                  for n, sz in self.sizes.items()}
        for stmt in dprog.body:
            match stmt:
                case VarDecl(_, vt, name, dsizes, _):
                    decls[name] = (vt, dsizes)
                case MultiDecl(mdecls):
                    for d in mdecls:
                        decls[d.name] = (d.vtype, d.sizes)

        # Let the simplifier see the declared type of each name
        for name, (vt, _) in decls.items():
            self.simplifier.env.define(name, vt)

        # Independent specialiser state per Jacobian column
        cols = [(arr, k, ik) for arr, shape in self._shapes.items()
                for k, ik in enumerate(product(*map(range, shape)))]
        cstate = {(arr, k): ColumnState() for arr, k, _ in cols}

        # Emit each tangent statement for every column next to its primal
        body = []
        for stmt in dprog.body:
            if _target_name(stmt) in self._tannames:
                for arr, k, ik in cols:
                    body += self._spec_stmt(stmt, arr, k, ik, cstate[arr, k])
            else:
                body.append(stmt)

        return Program(body)

    def generate_code(self, program):
        return self.codegen.generate(self.generate_tile_program(program))

    def _flatten_stmts(self, stmts):
        # Inline blocks, uniquely renaming their local declarations
        out = []
        for stmt in stmts:
            if isinstance(stmt, Block):
                inner = self._flatten_stmts(stmt.body)

                names = []
                for s in inner:
                    if isinstance(s, VarDecl):
                        names.append(s.name)
                    elif isinstance(s, MultiDecl):
                        names.extend(d.name for d in s.decls)

                sfx = f'__b{self._bidx}'
                self._bidx += 1

                renames = {n: Var(n + sfx) for n in names}
                out.extend(rename_vars(s, renames) for s in inner)
            elif isinstance(stmt, Region):
                raise ValueError('Cannot generate a Jacobian for a program '
                                 f'with a {stmt.kind} region')
            else:
                out.append(stmt)

        return out

    def _lower_array_inits(self, stmts):
        # Lower brace-initialised arrays into per-element writes
        out = []
        for stmt in stmts:
            match stmt:
                case VarDecl(c, vt, name, sizes, ArrayInit() as init) if sizes:
                    out.append(VarDecl(c, vt, name, sizes, None))
                    out += self._init_writes(name, sizes, init)
                case MultiDecl(decls):
                    if any(isinstance(d.init, ArrayInit) for d in decls):
                        for d in decls:
                            out += self._lower_array_inits([d])
                    else:
                        out.append(stmt)
                case _:
                    out.append(stmt)

        return out

    def _init_writes(self, name, sizes, init):
        if not all(isinstance(s, Int) for s in sizes):
            raise NotImplementedError('Non-constant sizes on initialised '
                                      f'array {name!r}')

        out = []
        for ik in product(*(range(s.value) for s in sizes)):
            # Zero-fill elements beyond the end of the initialiser
            val = init
            for i in ik:
                if isinstance(val, ArrayInit):
                    val = val.values[i] if i < len(val.values) else Int(0)
                elif val != Int(0):
                    raise NotImplementedError('Malformed initialiser on '
                                              f'array {name!r}')

            lhs = build_index(Var(name), [Int(i) for i in ik])
            out.append(ExprStmt(Assign(lhs, val)))

        return out

    def _validate(self, stmts):
        # Verify the program is straight-line and side-effect free
        for stmt in stmts:
            match stmt:
                case If() | While() | For() | DoWhile():
                    raise NotImplementedError('Control flow is not '
                                              'supported in tile generation')
                case VarDecl(_, _, _, _, init):
                    if init is not None:
                        _check_pure(init)
                case MultiDecl(decls):
                    for d in decls:
                        if d.init is not None:
                            _check_pure(d.init)
                case ExprStmt(Assign(lhs, rhs)):
                    _check_pure(rhs)
                    _, idxs = unwrap_index(lhs)
                    for ix in idxs:
                        _check_pure(ix)
                case ExprStmt(e) if e is not None:
                    _check_pure(e)

    def _spec_stmt(self, stmt, arr, k, ik, state):
        # Specialise one tangent statement for one Jacobian column
        self._state = state
        self._arr, self._k, self._kidx = arr, k, ik

        out = []
        match stmt:
            # Defer array tangent declarations to their first non-zero write
            case VarDecl(_, _, _, sizes, _) if sizes:
                pass
            # Scalar tangent declarations
            case VarDecl(c, vt, name, None, init):
                expr = self._sub_expr(init)
                if _is_zero(expr):
                    self._state.scalars.discard(name)
                else:
                    mn = self._mangle(name)
                    out.append(VarDecl(c, vt, mn, None, expr))
                    self._state.declared.add(name)
                    self._state.scalars.add(name)
            # Scalar tangent assignments
            case ExprStmt(Assign(Var(name), rhs)):
                expr = self._sub_expr(rhs)
                if _is_zero(expr):
                    self._state.scalars.discard(name)
                elif name in self._state.declared:
                    mn = self._mangle(name)
                    out.append(ExprStmt(Assign(Var(mn), expr)))
                    self._state.scalars.add(name)
                else:
                    vt = self._scalar_type(name)
                    mn = self._mangle(name)
                    out.append(VarDecl(False, vt, mn, None, expr))
                    self._state.declared.add(name)
                    self._state.scalars.add(name)
            # Tile array element assignments
            case ExprStmt(Assign(Index() as lhs, rhs)):
                name, idxs = unwrap_index(lhs)
                expr = self._sub_expr(rhs)
                if name in self._outtiles:
                    base = Var(f'{name}_{arr}')
                    nl = build_index(base, idxs + [Int(k)])
                    out.append(ExprStmt(Assign(nl, expr)))
                    if (ik := _int_key(idxs)) is not None:
                        self._state.outs.add((name, ik))
                else:
                    ik = _int_key(idxs)
                    if ik is None:
                        raise NotImplementedError('Non-constant write index '
                                                  f'on {name!r}')

                    # Overwritten elements stop resolving to their seed
                    self._state.written.add((name, ik))

                    if _is_zero(expr):
                        self._state.arrays.discard((name, ik))
                    else:
                        if name not in self._state.declared:
                            out.append(self._array_decl(name))
                            self._state.declared.add(name)

                        nl = build_index(Var(self._mangle(name)), idxs)
                        out.append(ExprStmt(Assign(nl, expr)))
                        self._state.arrays.add((name, ik))
            case _:
                raise NotImplementedError('Unsupported tangent statement: '
                                          f'{stmt}')

        return out

    def _mangle(self, name):
        # Mangle specialised names into a reserved namespace
        return f'{name}__{self._arr}_{self._k}'

    def _scalar_type(self, name):
        # Look up the type of the tangent's primal counterpart
        base = name.removeprefix(self.prefix)
        vt, _ = self._decls.get(base, ('fpdtype_t', None))

        return vt

    def _array_decl(self, name):
        base = name.removeprefix(self.prefix)
        vt, sizes = self._decls.get(base, ('fpdtype_t', None))
        if sizes is None:
            raise NotImplementedError('Unknown array sizes for tangent '
                                      f'array {name!r}')

        if not all(isinstance(s, Int) for s in sizes):
            raise NotImplementedError('Non-constant sizes on tangent '
                                      f'array {name!r}')

        return VarDecl(False, vt, self._mangle(name), sizes, None)

    def _sub_expr(self, expr):
        return self.simplifier.simplify_fully(self._sub(expr))

    def _sub(self, node):
        match node:
            case Index():
                name, idxs = unwrap_index(node)
                if name in self._seed_arrs:
                    ik = _int_key(idxs)
                    # Overwritten seed elements read their live tangent
                    if ik is not None and (name, ik) in self._state.written:
                        if (name, ik) in self._state.arrays:
                            return build_index(Var(self._mangle(name)), idxs)
                        else:
                            return Int(0)
                    else:
                        return self._sub_seed(name, idxs)
                elif name in self._outtiles:
                    # Redirect self-reads of written tile elements
                    ik = _int_key(idxs)
                    if ik is not None and (name, ik) in self._state.outs:
                        base = Var(f'{name}_{self._arr}')
                        return build_index(base, idxs + [Int(self._k)])
                    else:
                        raise NotImplementedError(f'Output tile {name!r} '
                                                  'read before written')
                elif name in self._tanarrs:
                    ik = _int_key(idxs)
                    if ik is None:
                        raise NotImplementedError('Non-constant read index '
                                                  f'on {name!r}')

                    if (name, ik) in self._state.arrays:
                        return build_index(Var(self._mangle(name)), idxs)
                    else:
                        return Int(0)
                else:
                    return map_ast(node, self._sub)
            case Var(name) if name in self._tannames:
                if name in self._state.scalars:
                    return Var(self._mangle(name))
                else:
                    return Int(0)
            case _:
                return map_ast(node, self._sub)

    def _sub_seed(self, name, idxs):
        # Seed arrays other than the active input with zero
        if self._seed_arrs[name] != self._arr:
            return Int(0)
        # Seed the active input with unity at the current column index
        else:
            shape, ik = self._shapes[self._arr], _int_key(idxs)
            if ik is None or len(ik) != len(shape):
                raise NotImplementedError(f'Bad seed access on {name!r}')

            return Int(1 if ik == self._kidx else 0)
