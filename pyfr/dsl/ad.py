from dataclasses import dataclass, field, replace
from math import prod
import re

from pyfr.dsl.derivs import Differentiator, build_index
from pyfr.dsl.nodes import (PRECISE_REGIONS, ArrayInit, Assign, Block,
                            DoWhile, ExprStmt, For, If, Index, Int, MultiDecl,
                            Number, Postfix, Program, Region, Stmt, Unary,
                            Var, VarDecl, While, expr_children, map_ast,
                            region_kinds, strip_regions, unwrap_index,
                            walk_ast)
from pyfr.dsl.simplifier import Simplifier
from pyfr.util import ndrange


def _active_part(node):
    # Drop the statements whose writes are not differentiated
    match node:
        case Region('nodiff', _, _):
            return Block([])
        case _:
            return map_ast(node, _active_part)


def _parse_ad_spec(args):
    # Sort the arguments of a region into 'y -> s', 'x <- s' and bare names
    prods, cons, free = {}, {}, []
    for a in args:
        if m := re.fullmatch(r'(\w+)\s*(->|<-)\s*(\w+)', a):
            bound = prods if m[2] == '->' else cons
            if m[1] in bound:
                raise ValueError(f'{m[1]!r} is bound twice in a '
                                 'differentiation region')

            bound[m[1]] = m[3]
        elif re.fullmatch(r'\w+', a):
            free.append(a)
        else:
            raise ValueError(f'Malformed differentiation argument {a!r}')

    return prods, cons, free


def _primal(name, env):
    # Declared type and constant extents of a name, empty for a scalar
    vtype, sizes = env.lookup_decl(name)
    if not sizes:
        return vtype, ()
    elif all(isinstance(s, Int) for s in sizes):
        return vtype, tuple(s.value for s in sizes)
    else:
        raise ValueError(f'Non-constant extents on array {name!r}')


def _check_storage(derivs, primals, env, kind):
    # Verify each derivative is declared with the type and extents required
    names = [s for s, _ in derivs]
    if len(set(names)) != len(names) or set(names) & set(primals):
        raise ValueError(f'{kind} region: derivative names {names} must be '
                         'distinct from each other and from the primals')

    for name, (vtype, shape) in derivs:
        if env.lookup_decl(name) is None:
            raise ValueError(f'{kind} region: {name!r} is not declared')
        if _primal(name, env) != (vtype, shape):
            raise ValueError(f'{kind} region: {name!r} must have the type '
                             f'{vtype} and extents {list(shape)}')


class TangentRegion:
    kind = 'tangent'

    def __init__(self, region, env):
        self.region, self.env = region, env

        prods, cons, free = _parse_ad_spec(region.args)
        if free or not prods or not cons:
            raise ValueError('Invalid tangent region arguments')

        # A name both read and written binds a single tangent
        both = [n for n in prods if n in cons and prods[n] != cons[n]]
        if both:
            raise ValueError(f'tangent region: {both} are bound to two '
                             'different tangents')

        self.outs, self.wrt, self.names = list(prods), list(cons), prods | cons
        if unknown := [n for n in self.names if env.lookup_decl(n) is None]:
            raise ValueError(f'Undeclared names {unknown} in tangent region')

        derivs = [(s, _primal(n, env)) for n, s in self.names.items()]
        _check_storage(derivs, self.names, env, self.kind)

    def lower(self):
        wrt, outs = self.wrt, self.outs
        gen = DerivativeGenerator(wrt, outs, self.names, self.env.scope())
        body = gen.transform(self.region.body)

        # Dependent names from outside the region must be listed as outputs
        deps = gen.dependent_vars | gen.dependent_arrays
        inner = {n.name for n in walk_ast(Program(body))
                 if isinstance(n, VarDecl)}
        if outside := sorted(deps - inner - set(wrt) - set(outs)):
            raise ValueError(f'{outside} depend on the inputs but are '
                             'declared outside the tangent region')

        return Block(body, comments=self.region.comments)


class JacobianRegion:
    kind = 'jacobian'

    def __init__(self, region, env):
        self.region, self.env = region, env

        prods, cons, free = _parse_ad_spec(region.args)
        if len(free) != 1 or cons or not prods:
            raise ValueError('Invalid jacobian region arguments')

        self.y, self.tiles = y, tiles = free[0], prods
        if unknown := [n for n in [y, *tiles] if env.lookup_decl(n) is None]:
            raise ValueError(f'Undeclared names {unknown} in jacobian region')
        if y in tiles:
            raise ValueError(f'Jacobian output {y!r} can not be an input')

        self.shapes = shapes = {x: _primal(x, env)[1] for x in tiles}
        if scalars := [x for x in tiles if not shapes[x]]:
            raise ValueError(f'Jacobian inputs {scalars} must be arrays')

        # Each tile has the extents of the output plus the entries of its input
        vtype, shape = _primal(y, env)
        derivs = [(t, (vtype, shape + (prod(shapes[x]),)))
                  for x, t in tiles.items()]
        _check_storage(derivs, [y, *tiles], env, self.kind)

    def lower(self):
        # Specialise the tangent of the straight-line body for each direction
        spec = DirectionSpecialiser(self.shapes, self.y, self.tiles, self.env)
        body = spec.specialise(self.region.body)

        return Block(body, comments=self.region.comments)


def lower_ad_regions(prog, env):
    # Replace the differentiation regions by their derivative bodies
    def visit(node, env):
        match node:
            case Region('tangent', _, _):
                return TangentRegion(node, env).lower()
            case Region('jacobian', _, _):
                return JacobianRegion(node, env).lower()
            case Program(body) | Block(body) | Region(_, body, _):
                inner, nbody = env.scope(), []
                for s in body:
                    nbody.append(visit(s, inner))
                    inner.declare(s)

                return replace(node, body=nbody)
            case If(_, then, else_):
                nelse = visit(else_, env.scope()) if else_ else None
                return replace(node, then=visit(then, env.scope()),
                               else_=nelse)
            case For(init, _, _, body):
                inner = env.scope()
                inner.declare(init)
                return replace(node, body=visit(body, inner))
            case While(_, body) | DoWhile(body, _):
                return replace(node, body=visit(body, env.scope()))
            case _:
                return node

    if not region_kinds(prog).isdisjoint({'tangent', 'jacobian'}):
        prog = visit(prog, env)

        # Simplify the differentiated program without IEEE constraints
        prog = Simplifier(env).simplify_program(prog)

    # Strip any nodiff regions
    return strip_regions(prog, {'nodiff'})


def _check_pure(expr):
    # Reject expressions whose side effects the transform cannot see
    match expr:
        case Assign() | Postfix() | Unary('++' | '--', _):
            raise NotImplementedError('Side effects inside expressions are '
                                      'not differentiable')
        case _:
            for c in expr_children(expr):
                _check_pure(c)


class DerivativeGenerator:
    def __init__(self, wrt, outs, names, env):
        self.wrt = set(wrt)
        self.outs = set(outs)
        self.names = dict(names)
        self.dependent_vars = set()
        self.dependent_arrays = set()
        self.simplifier = Simplifier(env)

        # Nesting depth of nodiff regions being transformed
        self._inactive = 0

        # Differentiate along every input at once
        self.diff = Differentiator(self.wrt, self.dependent_vars,
                                   self.dependent_arrays, self.tangent_name)

    def tangent_name(self, name):
        # Bound tangents keep their names, those of other names are private
        return self.names.get(name, f'_d_{name}')

    def _analyse(self, stmts):
        # Gather the target and value of every write in the program
        writes = []
        for node in walk_ast(Program(stmts)):
            # Let the simplifier see the declared type of each name
            if isinstance(node, VarDecl):
                self.simplifier.env.declare(node)

        for node in walk_ast(_active_part(Program(stmts))):
            match node:
                case VarDecl(_, _, name, sizes, init):
                    if init is not None:
                        writes.append((name, sizes is not None, init))
                case Assign(lhs, rhs):
                    name, indices = unwrap_index(lhs)
                    if name is not None:
                        writes.append((name, bool(indices), rhs))

        # Propagate the dependence on the inputs through the writes
        deps, darrs = self.dependent_vars, self.dependent_arrays
        ndeps = -1
        while ndeps != len(deps) + len(darrs):
            ndeps = len(deps) + len(darrs)
            for name, is_array, rhs in writes:
                if self.diff.contains_var(rhs):
                    if is_array:
                        darrs.add(name)
                    else:
                        deps.add(name)

    def _dependent(self, name):
        return name in self.dependent_vars or name in self.dependent_arrays

    def _carries(self, name):
        # Inputs and outputs carry a tangent even when independent
        forced = name in self.outs or name in self.wrt

        return forced or self._dependent(name)

    def _diff_and_simplify(self, expr):
        return self.simplifier.simplify_fully(self.diff.diff(expr))

    def _tangent_of(self, expr):
        # Values written inside a nodiff region are constants
        if self._inactive:
            return Int(0)
        else:
            return self._diff_and_simplify(expr)

    def _transform_stmt(self, stmt):
        match stmt:
            # Multi-declarations are flattened and handled individually
            case MultiDecl(decls):
                return [s for d in decls for s in self._transform_stmt(d)]
            # Variable and array declarations
            case VarDecl(_, vtype, name, sizes, init):
                return self._transform_decl(stmt, vtype, name, sizes, init)
            # Assignments to variables and array entries
            case ExprStmt(Assign(Var() | Index() as lhs, expr)):
                return self._transform_assign(stmt, lhs, expr)
            # Block statements
            case Block(body):
                stmts = [s for st in body for s in self._transform_stmt(st)]
                return [Block(stmts)]
            # Conditional statements
            case If(cond, then, else_):
                _check_pure(cond)

                def tf(s): return self._wrap_block(self._transform_stmt(s))

                return [If(cond, tf(then), tf(else_) if else_ else None)]
            # Loop statements
            case While(cond, body):
                tbody = self._wrap_block(self._transform_stmt(body))
                return [While(cond, tbody)]
            case For(init, cond, step, body):
                tbody = self._wrap_block(self._transform_stmt(body))
                return [For(init, cond, step, tbody)]
            case DoWhile(body, cond):
                tbody = self._wrap_block(self._transform_stmt(body))
                return [DoWhile(tbody, cond)]
            # Writes inside a nodiff region are treated as constants
            case Region('nodiff', body, _):
                return self._transform_nodiff(stmt, body)
            # Differentiate region bodies, preserving the wrapper
            case Region(kind, body, args):
                if kind in PRECISE_REGIONS:
                    raise ValueError(f'Cannot differentiate a {kind} region')
                if kind in {'tangent', 'jacobian'}:
                    raise ValueError(f'Cannot nest a {kind} region inside a '
                                     'differentiation region')

                stmts = [s for st in body for s in self._transform_stmt(st)]
                return [Region(kind, stmts, args)]
            # Increments inside a nodiff region leave a constant behind
            case ExprStmt(Postfix(_, e) | Unary('++' | '--', e)):
                name, idxs = unwrap_index(e)
                for ix in idxs:
                    _check_pure(ix)

                if self._inactive and name and self._carries(name):
                    dlhs = build_index(Var(self.tangent_name(name)), idxs)
                    return [ExprStmt(Assign(dlhs, Int(0))), stmt]
                else:
                    return [stmt]
            # Other expression statements may not write through their operands
            case ExprStmt(e) if e is not None:
                for c in expr_children(e):
                    _check_pure(c)

                return [stmt]
            # Pass through other statements unchanged
            case _:
                return [stmt]

    def _transform_decl(self, stmt, vtype, name, sizes, init):
        if init is not None:
            _check_pure(init)

        if sizes and not all(isinstance(s, Int) for s in sizes):
            raise ValueError(f'Non-constant extents on array {name!r}')

        # Array initialisers depending on the inputs are differentiated
        dinit_arr = sizes and init and self.diff.contains_var(init)

        out = [stmt]
        if self._dependent(name):
            # Initialise the tangent from the initialiser, else to zero
            if dinit_arr:
                dinit = self._diff_init(init)
            elif sizes:
                dinit = ArrayInit([Int(0)])
            elif init:
                dinit = self._diff_and_simplify(init)
            else:
                dinit = Int(0)

            dname = self.tangent_name(name)
            out.append(VarDecl(False, vtype, dname, sizes, dinit))

        return out

    def _diff_init(self, init):
        # Differentiate each entry of a nested array initialiser
        vals = []
        for v in init.values:
            if isinstance(v, ArrayInit):
                vals.append(self._diff_init(v))
            else:
                vals.append(self._diff_and_simplify(v))

        return ArrayInit(vals)

    def _transform_assign(self, stmt, lhs, expr):
        name, indices = unwrap_index(lhs)
        for e in [expr, *indices]:
            _check_pure(e)

        # Force a tangent for writes to the inputs and outputs
        if name is not None and self._carries(name):
            dlhs = build_index(Var(self.tangent_name(name)), indices)
            tangents = [ExprStmt(Assign(dlhs, self._tangent_of(expr)))]
        else:
            tangents = []

        # Tangents precede the write so they read pre-update values
        return tangents + [stmt]

    def _transform_nodiff(self, stmt, body):
        # Copy the body, zeroing the tangents of the names it writes
        self._inactive += 1
        stmts = [s for st in body for s in self._transform_stmt(st)]
        self._inactive -= 1

        return [Block(stmts, comments=stmt.comments)]

    def _wrap_block(self, stmts):
        if len(stmts) == 1:
            return stmts[0]
        else:
            return Block(stmts)

    def transform(self, stmts):
        self._analyse(stmts)

        return [s for st in stmts for s in self._transform_stmt(st)]


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


def _init_entries(sizes, init):
    out = []
    for ik in ndrange(*(s.value for s in sizes)):
        # Zero-fill entries beyond the end of the initialiser
        val = init
        for i in ik:
            if isinstance(val, ArrayInit):
                val = val.values[i] if i < len(val.values) else Int(0)

        out.append((ik, val))

    return out


@dataclass(slots=True)
class _Hole(Stmt):
    # Placeholder for tangent declarations hoisted out of nested blocks
    decls: dict = field(default_factory=dict)

    def add(self, decl):
        self.decls[decl.name] = decl


@dataclass(slots=True)
class DirectionState:
    # Live tangent entries, keyed by name and constant index
    live: set = field(default_factory=set)

    # Out tile entries written and seed entries overwritten
    outs: set = field(default_factory=set)
    written: set = field(default_factory=set)

    # Tangent declarations made for the direction
    declared: set = field(default_factory=set)

    # Declaration holes of the direction, with the block depth of each
    holes: dict = field(default_factory=dict)

    def forget(self, names):
        # Drop the tangents of names which have gone out of scope
        self.declared -= names
        self.live = {(n, ik) for n, ik in self.live if n not in names}
        self.holes = {n: h for n, h in self.holes.items() if n not in names}

    def clone(self):
        return DirectionState(set(self.live), set(self.outs),
                              set(self.written), set(self.declared),
                              dict(self.holes))

    def merge(self, other):
        # Combine the states at the end of the two paths of a conditional
        return DirectionState(self.live | other.live, self.outs | other.outs,
                              self.written | other.written,
                              self.declared | other.declared,
                              self.holes | other.holes)


class DirectionSpecialiser:
    def __init__(self, shapes, out, tiles, env):
        self._shapes = dict(shapes)
        self._out = out

        # Tile receiving the derivatives with respect to each input
        self._tiles = dict(tiles)

        # Declarations visible at the current statement
        self.env = env.scope()
        self.simplifier = Simplifier(self.env)

        # Forward mode transform whose tangent is specialised per direction
        self._gen = DerivativeGenerator(tiles, [out], {}, env.scope())

    def specialise(self, stmts):
        gen = self._gen
        dstmts = gen.transform(stmts)

        # Map the tangent names of the transform back to their primals
        deps = set(self._shapes).union([self._out], gen.dependent_vars,
                                       gen.dependent_arrays)
        self._tannames = {gen.tangent_name(n): n for n in deps}
        self._seed_arrs = {gen.tangent_name(n): n for n in self._shapes}
        self._outtiles = {gen.tangent_name(self._out): self._out}

        # Independent specialiser state per Jacobian direction
        self._dirs = [(arr, k, ik) for arr, shape in self._shapes.items()
                      for k, ik in enumerate(ndrange(*shape))]
        self._kidxs = {(arr, k): ik for arr, k, ik in self._dirs}
        self._dstate = {(arr, k): DirectionState()
                        for arr, k, _ in self._dirs}

        # Nesting depth of the block being specialised
        self._depth = 0

        # Depths of the enclosing conditional statements
        self._ifdepths = []

        # Declarations of tangents whose primal is declared outside the body
        self._top = _Hole()

        return self._prune([self._top] + self._spec_stmts(dstmts))

    def _select(self, arr, k):
        # Make the given Jacobian direction the one being specialised
        self._arr, self._k, self._kidx = arr, k, self._kidxs[arr, k]
        self._state = self._dstate[arr, k]

    def _spec_stmts(self, stmts):
        # Emit each tangent statement for every direction next to its primal
        body = []
        for stmt in stmts:
            match stmt:
                case Block(inner):
                    body.append(self._spec_block(stmt, inner))
                    continue
                case If(cond, then, else_):
                    body += self._spec_if(cond, then, else_)
                    continue
                case For() | While() | DoWhile() | Region():
                    raise NotImplementedError('Loops and regions are not '
                                              'supported in tile generation')
                case VarDecl():
                    self._declare(stmt)

            if _target_name(stmt) in self._tannames:
                for arr, k, ik in self._dirs:
                    body += self._spec_stmt(stmt, arr, k)
            else:
                body.append(stmt)

        return body

    def _spec_block(self, block, inner):
        # Specialise a nested block, forgetting its locals on exit
        outer, self.env = self.env, self.env.scope()
        self.simplifier.env = self.env
        self._depth += 1
        body = self._spec_stmts(inner)
        self._depth -= 1
        local = set(self.env.names)
        self.env = self.simplifier.env = outer

        for state in self._dstate.values():
            state.forget(local)

        return Block(body, comments=block.comments)

    def _spec_if(self, cond, then, else_):
        # Specialise each path from the same state, then reconcile them
        self._ifdepths.append(self._depth)
        before = {key: s.clone() for key, s in self._dstate.items()}

        tblock = self._spec_branch(then)
        tstates, self._dstate = self._dstate, before
        eblock = self._spec_branch(else_) if else_ else Block([])

        for key in self._dstate:
            self._dstate[key] = self._merge(key, tstates[key],
                                            self._dstate[key], tblock, eblock)

        self._ifdepths.pop()

        return [If(cond, tblock, eblock if eblock.body else None)]

    def _spec_branch(self, branch):
        # Specialise a branch of a conditional as a nested block
        block = branch if isinstance(branch, Block) else Block([branch])

        return self._spec_block(block, block.body)

    def _merge(self, key, tstate, estate, tblock, eblock):
        # Reconcile the direction state after the two paths of a conditional
        self._arr, self._k = key
        self._kidx = self._kidxs[key]

        # Seeds overwritten on one path are materialised on the other
        for name, ik in tstate.written ^ estate.written:
            if (name, ik) in tstate.written:
                self._seed_entry(estate, eblock, name, ik)
            else:
                self._seed_entry(tstate, tblock, name, ik)

        # Tangents live on one path only are zeroed on the other
        for name, ik in tstate.live ^ estate.live:
            if (name, ik) in tstate.live:
                self._zero_tangent(estate, eblock, name, ik)
            else:
                self._zero_tangent(tstate, tblock, name, ik)

        # Tile entries written on one path only are zeroed on the other
        for name, ik in tstate.outs ^ estate.outs:
            block = eblock if (name, ik) in tstate.outs else tblock
            lhs = self._tile(name, [Int(i) for i in ik])
            block.body.append(ExprStmt(Assign(lhs, Int(0))))

        return tstate.merge(estate)

    def _seed_entry(self, state, block, name, ik):
        # Store the seed value of an entry left untouched on this path
        self._state = state
        idxs = [Int(i) for i in ik]
        seed = self._sub_seed(name, idxs)
        block.body += self._spec_write(name, idxs, ik, seed, self._depth)

    def _zero_tangent(self, state, block, name, ik):
        # Zero a tangent on the path where it is not live, if it may be stale
        if name in state.declared:
            lhs = build_index(Var(self._mangle(name)), [Int(i) for i in ik])
            block.body.append(ExprStmt(Assign(lhs, Int(0))))

    def _declare(self, stmt):
        # Reject declarations which shadow a name of an enclosing scope
        if self.env.lookup_decl(stmt.name) is not None:
            raise NotImplementedError(f'Declaration of {stmt.name!r} shadows '
                                      'an enclosing declaration')

        self.env.declare(stmt)

    def _prune(self, stmts):
        # Fill the declaration holes which were needed and drop the rest
        out = []
        for stmt in stmts:
            match stmt:
                case Block(body):
                    out.append(Block(self._prune(body),
                                     comments=stmt.comments))
                case If(cond, then, else_):
                    nelse = self._prune([else_])[0] if else_ else None
                    out.append(If(cond, self._prune([then])[0], nelse))
                case _Hole(decls):
                    out += decls.values()
                case _:
                    out.append(stmt)

        return out

    def _hole(self, name):
        # Leave a hole should the tangent first be written in a nested block
        hole = _Hole()
        self._state.holes[name] = (hole, self._depth)

        return hole

    def _hoist(self, name, array=False, maxdepth=None):
        # Declare a tangent first written in a nested block at its hole
        maxdepth = self._depth - 1 if maxdepth is None else maxdepth
        hole, depth = self._state.holes.get(name, (None, None))

        # Tangents of names declared outside the body are hoisted to the top
        if hole is None and maxdepth >= 0:
            hole, depth = self._top, 0

        if hole is not None and depth <= maxdepth:
            # Tangents first written under a branch start from zero
            if any(d >= depth for d in self._ifdepths):
                init = ArrayInit([Int(0)]) if array else Int(0)
            else:
                init = None

            hole.add(self._tangent_decl(name, array, init))
            self._state.declared.add(name)
            return True
        else:
            return False

    def _spec_stmt(self, stmt, arr, k):
        # Specialise one tangent statement for one Jacobian direction
        self._select(arr, k)

        match stmt:
            # Tangent declarations, with their non-zero initial entries
            case VarDecl(_, _, name, sizes, init):
                if sizes is None:
                    entries = [((), init)]
                elif isinstance(init, ArrayInit):
                    entries = _init_entries(sizes, init)
                else:
                    entries = []

                out = [self._hole(name)]
                for ik, val in entries:
                    idxs = [Int(i) for i in ik]
                    expr = self._sub_expr(val)
                    out += self._spec_write(name, idxs, ik, expr)

                return out
            # Tangent and tile assignments
            case ExprStmt(Assign(lhs, rhs)):
                name, idxs = unwrap_index(lhs)
                expr, ik = self._sub_expr(rhs), _int_key(idxs)
                if name in self._outtiles:
                    return self._spec_tile_write(name, idxs, ik, expr)
                elif ik is None:
                    raise NotImplementedError('Non-constant write index '
                                              f'on {name!r}')
                else:
                    return self._spec_write(name, idxs, ik, expr)
            case _:
                raise NotImplementedError('Unsupported tangent statement: '
                                          f'{stmt}')

    def _spec_tile_write(self, name, idxs, ik, expr):
        # Write the tile entry for the direction, tracking constant entries
        nl = self._tile(name, idxs)
        if ik is not None:
            self._state.outs.add((name, ik))

        # Skip the identity writes of accumulations with no tangent
        if expr != nl:
            return [ExprStmt(Assign(nl, expr))]
        else:
            return []

    def _spec_write(self, name, idxs, ik, expr, maxdepth=None):
        # Overwritten seed entries stop resolving to their seed
        if name in self._seed_arrs:
            self._state.written.add((name, ik))

        if _is_zero(expr):
            self._state.live.discard((name, ik))
            return []

        # Declare the tangent at its hole, else alongside its first write
        nl = build_index(Var(self._mangle(name)), idxs)
        if name in self._state.declared or self._hoist(name, bool(idxs),
                                                       maxdepth):
            # Skip the identity writes of accumulations with no tangent
            out = [ExprStmt(Assign(nl, expr))] if expr != nl else []
        elif idxs:
            out = [self._tangent_decl(name, True), ExprStmt(Assign(nl, expr))]
        else:
            out = [self._tangent_decl(name, init=expr)]

        self._state.declared.add(name)
        self._state.live.add((name, ik))

        return out

    def _mangle(self, name):
        # Mangle specialised names into a reserved namespace
        return f'{name}__{self._arr}_{self._k}'

    def _tile(self, name, idxs):
        # Index the entry of the output tile for the current direction
        tile = Var(self._tiles[self._arr])

        return build_index(tile, idxs + [Int(self._k)])

    def _tangent_decl(self, name, array=False, init=None):
        # Declare the tangent temporary with the type and extents of its primal
        vt, sizes = self.env.lookup_decl(self._tannames[name])
        return VarDecl(False, vt, self._mangle(name), sizes, init)

    def _sub_expr(self, expr):
        return self.simplifier.simplify_fully(self._sub(expr))

    def _sub(self, node):
        match node:
            case Var() | Index():
                name, idxs = unwrap_index(node)
                ik = _int_key(idxs)
                if name in self._seed_arrs:
                    # Overwritten seed entries read their live tangent
                    if ik is not None and (name, ik) in self._state.written:
                        return self._live_read(name, idxs, ik)
                    else:
                        return self._sub_seed(name, idxs)
                elif name in self._outtiles:
                    # Redirect self-reads of written tile entries
                    if ik is not None and (name, ik) in self._state.outs:
                        return self._tile(name, idxs)
                    else:
                        raise NotImplementedError(f'Output tile {name!r} '
                                                  'read before written')
                elif name in self._tannames:
                    if ik is None:
                        raise NotImplementedError('Non-constant read index '
                                                  f'on {name!r}')

                    return self._live_read(name, idxs, ik)
                else:
                    return map_ast(node, self._sub)
            case _:
                return map_ast(node, self._sub)

    def _live_read(self, name, idxs, ik):
        # Read a live tangent entry, with dead entries reading as zero
        if (name, ik) in self._state.live:
            return build_index(Var(self._mangle(name)), idxs)
        else:
            return Int(0)

    def _sub_seed(self, name, idxs):
        # Seed arrays other than the active input with zero
        if self._seed_arrs[name] != self._arr:
            return Int(0)
        # Seed the active input with unity at the current direction index
        else:
            shape, ik = self._shapes[self._arr], _int_key(idxs)
            if ik is None or len(ik) != len(shape):
                raise NotImplementedError(f'Bad seed access on {name!r}')

            return Int(1 if ik == self._kidx else 0)
