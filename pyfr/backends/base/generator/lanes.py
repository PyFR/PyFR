from dataclasses import replace
from functools import reduce
import itertools as it
from math import prod

from pyfr.backends.base.generator.reduce import ScratchSlots, butterfly
from pyfr.dsl.nodes import (Assign, Binary, Block, Call, Cast, Comment,
                            DSLCall, DSLVar, ExprStmt, If, Index, Int,
                            MultiDecl, Postfix, Region, Transformer, Unary,
                            Var, VarDecl, unwrap_index, walk_ast)
from pyfr.dsl.rewriter import parse_expr, parse_program, rename_vars


def _written(expr):
    # Find the array or variable an expression assigns to, if any
    match expr:
        case (Assign(target) | Postfix('++' | '--', target) |
              Unary('++' | '--', target)):
            return unwrap_index(target)[0]
        case _:
            return None


def _reject_coop(ast):
    # Reject what only the lanes of a cooperative kernel can make use of
    for n in walk_ast(ast):
        match n:
            case DSLVar('lidx' | 'lsz' as name):
                raise ValueError(f'@{name} requires a coop kernel')
            case (DSLCall('sync' | 'load' | 'store' as r, _) |
                  Region('allreduce' as r, _, _)):
                raise ValueError(f'{r} requires a coop kernel')


def _copied(stmt):
    # Find the kind, target and expression of a top-level load or store
    match stmt:
        case ExprStmt(DSLCall('load' | 'store' as kind, [Var(name), expr])):
            return kind, name, expr
        case _:
            return None


def has_shared(ast):
    return any(isinstance(n, VarDecl) and n.shared for n in walk_ast(ast))


def shared_decls(stmt):
    # Find the declarations of shared storage made by a statement
    match stmt:
        case VarDecl(shared=True):
            return [stmt]
        case MultiDecl([VarDecl(shared=True), *_] as ds):
            return ds
        case _:
            return []


def body_scope(ast):
    # Find the node which holds the statements of the body
    match ast.body:
        case [Block() as blk]:
            return blk
        case _:
            return ast


def with_body_scope(ast, body):
    # Replace the statements of the body scope of the program
    match ast.body:
        case [Block() as blk]:
            return replace(ast, body=[replace(blk, body=body)])
        case _:
            return replace(ast, body=body)


def make_lanemodel(gen, spec):
    # Assign lanes to each item of the iteration range
    if spec is not None:
        return LaneGroup(gen, spec)
    elif gen.ndim == 2:
        return gen.rowlanes_cls(gen)
    else:
        return SingleLane(gen)


def unravel_index(flat, cdims):
    # Unravel the flat index of an entry into the indices of the array
    ixs = [flat]
    for d in reversed(cdims[1:]):
        ixs = [f'({ixs[0]} / {d})', f'{ixs[0]} % {d}', *ixs[1:]]

    return [f'[{i}]' for i in ixs]


def array_names(expr):
    # Collect the names of the whole arrays an entry-wise expression combines
    names = []
    for n in walk_ast(expr):
        match n:
            case Var(name):
                names.append(name)
            case DSLVar() | DSLCall() | Index():
                raise ValueError('Expected an expression of whole arrays')

    return names


def array_sources(expr, args, shared):
    # Look up the extents and type of each array an expression combines
    srcs = {}
    for name in array_names(expr):
        arg = args.get(name)

        if name in shared:
            srcs[name] = shared[name].cdims, shared[name].dtype
        elif arg is None or arg.isview or arg.ismpi or arg.isreduce:
            raise ValueError(f'{name!r} can not be read by a load or store')
        else:
            srcs[name] = arg.cdims, arg.dtype

    return srcs


class SharedDecl:
    def __init__(self, decl, args):
        self.name = decl.name
        self.dtype = decl.vtype

        # Arguments whose entries are loaded into the storage, if any
        self.load = None

        match decl.init:
            case DSLCall('load', [expr]):
                self._init_load(decl, expr, args)
            case DSLCall('load', _):
                raise ValueError(f'Shared {self.name!r} must load an '
                                 'expression of arguments')
            case None:
                self._init_sized(decl)
            case _:
                raise ValueError(f'Shared {self.name!r} cannot be initialised')

        self.cdimstr = ''.join(f'[{d}]' for d in self.cdims)

    def nbytes(self, gen):
        return prod(self.cdims)*gen.npdtype(self.dtype).itemsize

    def _init_sized(self, decl):
        # Validation
        if not decl.sizes or not all(isinstance(s, Int) for s in decl.sizes):
            raise ValueError(f'Shared {self.name!r} needs constant extents')

        # Dimensions
        self.cdims = [int(s.value) for s in decl.sizes]

    def _init_load(self, decl, expr, args):
        # Validation
        if decl.sizes:
            raise ValueError(f'Shared {self.name!r} takes its extents from '
                             'the arguments it loads')

        # Collect the arguments of the expression in order of appearance
        self.expr, self.srcs = expr, array_sources(expr, args, {})
        self.load = [args[n] for n in self.srcs]

        # Take the shape of the widest argument for the rest to broadcast over
        wide = max(self.load, key=lambda a: a.ncdim, default=None)
        if not wide or not wide.cdims:
            raise ValueError(f'Shared {self.name!r} loads no array')
        self.cdims = list(wide.cdims)

        # Take the type given, else of a single argument, else fpdtype_t
        if decl.vtype is None:
            self.dtype = wide.dtype if isinstance(expr, Var) else 'fpdtype_t'
        bf16 = any(a.dtype == 'bf16' for a in self.load)
        if bf16 and self.dtype == 'bf16' and not isinstance(expr, Var):
            raise ValueError(f'Shared {self.name!r} cannot mix bf16 entries')


class StoreStmt:
    def __init__(self, call, args, shared):
        match call:
            case DSLCall('store', [Var(aname), expr]):
                self.arg, self.expr = self._target(aname, args), expr
            case _:
                raise ValueError('Stores must be statements of an argument '
                                 'and an expression')

        # Name, extents and type of the argument stored to
        arg = self.arg
        self.name, self.cdims, self.dtype = arg.name, arg.cdims, arg.dtype

        # Look up the extents and type of each array the expression combines
        self.srcs = array_sources(expr, args, shared)

    def _target(self, aname, args):
        # Check the target is a per-item array argument of the kernel
        if not (arg := args.get(aname)):
            raise ValueError(f'Store to an unknown argument {aname!r}')
        if arg.intent == 'in' or arg.attrs:
            raise ValueError(f'Argument {aname!r} can not be stored')
        if not arg.cdims:
            raise ValueError(f'Stored argument {aname!r} must be an array')

        return arg


class GroupIndexing:
    def __init__(self, gen, size, groups):
        self.gen, self.size, self.groups = gen, size, groups

        # Have the lanes of the group walk the entries of its item
        self.start, self.entry = '@lidx', '_li'
        self.extent = self.group = self.ritem = self.witem = self.item = ''

    @property
    def prologue(self):
        return f'ixdtype_t _x = {self.gen._grp};'

    def lidx(self):
        return Cast('int', Var(self.gen._lid[0]))

    def index_by_group(self, node, names):
        return node

    def if_active(self, stmt):
        return stmt


class GroupsIndexing(GroupIndexing):
    def __init__(self, gen, size, groups):
        super().__init__(gen, size, groups)

        # Have the lanes of the block walk the entries of its items in turn
        self.start, self.entry = '_bl', f'(_li / {groups})'
        self.extent, self.group = f'[{groups}]', f'[_li % {groups}]'

        # Clamp the groups past the last item onto it or skip their writes
        self.ritem = (f'ixdtype_t _xg = _xb + _li % {groups}; '
                      'ixdtype_t _x = (_xg < _nx) ? _xg : _nx - 1;')
        self.witem = f'ixdtype_t _x = _xb + _li % {groups}; if (_x < _nx)'

        # Have each lane-group of the block take an item of the range
        self.item = '''
            ixdtype_t _x = min(_xb + _g, _nx - 1);
            const int _active = _xb + _g < _nx;'''

    @property
    def prologue(self):
        gen = self.gen

        return f'''
            const int _bl = (int) {gen._lid[0]};
            const int _g = {gen._group_index(self.size)};
            ixdtype_t _xb = {gen._grp}*{self.groups};'''

    def lidx(self):
        return Binary('%', Var('_bl'), Int(self.size))

    def index_by_group(self, node, names):
        def index(node):
            match node:
                case Var(name) if name in names:
                    return Index(node, Var('_g'))
                case DSLCall('load' | 'store', _):
                    return node
                case _:
                    return node.map(index)

        # Index the copy of the storage belonging to the lane-group
        return index(node)

    def if_active(self, stmt):
        # Keep the groups clamped onto the last item from storing to it
        guarded = replace(stmt, comments=())

        return If(Var('_active'), guarded, comments=stmt.comments)


class LaneLowering(Transformer):
    def __init__(self, lanemodel, subs, env):
        super().__init__(env)
        self.lanemodel, self.subs = lanemodel, subs

    def visit(self, node):
        model, subs = self.lanemodel, self.subs

        match node:
            case DSLVar(name) if name in subs:
                return subs[name]
            # Drop the empty statements left by barriers of one lane
            case ExprStmt(None, comments=cs):
                return [Comment(*c) for c in cs]
            case ExprStmt(DSLCall('sync', []), comments=cs):
                stmt = model.sync_stmt(cs)
                return self.visit(stmt) if stmt.expr is None else stmt
            case DSLCall('sync', _):
                raise ValueError('Invalid synchronisation scope')
            case ExprStmt(DSLCall('load', [Var(name), _]), comments=cs):
                return model.copy_stmt(model.shared[name], True, cs)
            case DSLCall('load', _):
                raise ValueError('Loads must declare shared storage')
            case ExprStmt(DSLCall('store', _) as call, comments=cs):
                store = StoreStmt(call, model.args, model.shared)
                return model.copy_stmt(store, False, cs)
            case Region('allreduce', body, names, comments=cs):
                red = model.allreduce(list(names), body, self.env, cs)
                return self.visit(red)
            case DSLCall('partner', _):
                raise ValueError('@partner must be inside an allreduce region')
            case ExprStmt(expr) if model.stores_arg(expr):
                return model.indexing.if_active(super().visit(node))
            case _:
                return super().visit(node)


class BaseLaneModel:
    coop = False

    def __init__(self, gen):
        self.gen = gen
        self.size = self.groups = 1
        self.decls = []
        self.shared = {}
        self.prologue = []
        self.staged = set()

        # Whether the barriers of the lanes also order global memory
        self.fenced = False

    def _split_shared(self, ast, args):
        body, decls = [], []

        # Separate the top-level shared declarations from the body
        for stmt in body_scope(ast).body:
            calls = [n for n in walk_ast(stmt) if isinstance(n, DSLCall)]

            if new := [SharedDecl(d, args) for d in shared_decls(stmt)]:
                decls += new
            elif any(c.name == 'load' for c in calls):
                raise ValueError('Loads must declare shared storage')
            else:
                body.append(stmt)

            # Leave a load of the storage in place of its declaration
            for d in new:
                if d.load is not None:
                    load = DSLCall('load', [Var(d.name), d.expr])
                    body.append(ExprStmt(load, comments=stmt.comments))
        ast = with_body_scope(ast, body)

        # Validation
        if has_shared(ast):
            raise ValueError('Shared declarations must be at the top level')
        if any(d.name in args for d in decls):
            raise ValueError('Shared declarations can not shadow arguments')
        if len({d.name for d in decls}) != len(decls):
            raise ValueError('Shared declarations must have distinct names')

        return ast, decls

    def extents(self, decl):
        # Declare the storage with the extents of its declaration
        return decl.cdimstr

    def validate_args(self, vargs):
        pass

    def check_storage(self, ast):
        return ast


class SingleLane(BaseLaneModel):
    def hoist_shared(self, ast, args):
        if has_shared(ast):
            raise ValueError('Shared declarations require a coop kernel')

        return ast

    def lower(self, ast, env):
        _reject_coop(ast)

        return ast


class RowLanes(BaseLaneModel):
    def hoist_shared(self, ast, args):
        body = body_scope(ast).body

        # Validation
        lead = len(list(it.takewhile(shared_decls, body)))
        if any(has_shared(s) for s in body[lead:]):
            raise ValueError('Shared declarations must lead a 2D kernel body')

        # Separate the shared declarations leading the body
        ast, decls = self._split_shared(ast, args)
        for d in decls:
            self._check_staged(d)
        self.staged = {d.load[0].name for d in decls}
        self.columns = {d.name: d for d in decls}

        # Drop their loads, which the block carries out before its rows
        return with_body_scope(ast, body_scope(ast).body[len(decls):])

    def _check_staged(self, decl):
        # Check a bare column argument of the type of the storage is loaded
        if decl.load is None or not isinstance(decl.expr, Var):
            raise ValueError(f'Shared {decl.name!r} needs a column argument')

        arg = decl.load[0]
        if not arg.isbroadcastc:
            raise ValueError(f'Staged {arg.name!r} must be broadcast-col')
        if decl.dtype != arg.dtype:
            raise ValueError(f'Shared {decl.name!r} must be of the type of '
                             f'{arg.name!r}')

    def lower(self, ast, env):
        _reject_coop(ast)
        decls = self.columns

        def index(node):
            # Reject writes to the staged storage
            if _written(node) in decls:
                raise ValueError(f'Shared {_written(node)!r} is staged and '
                                 'can not be written')

            # Lower the reads of the staged storage
            name, ixs = unwrap_index(node)
            if name in decls:
                if len(ixs) != len(decls[name].cdims):
                    raise ValueError(f'Shared {name!r} must be fully indexed')

                return self._read(decls[name], [index(i) for i in ixs])
            else:
                return node.map(index)

        return index(ast)

    def _read(self, decl, ixs):
        # Read the argument in place of the storage
        return reduce(Index, ixs, decl.expr)


class StagedRowLanes(RowLanes):
    def __init__(self, gen):
        super().__init__(gen)

        # Items of a block, one per column, and the lanes splitting their rows
        self.cols, self.rows = gen.block2d[:2]

    def hoist_shared(self, ast, args):
        ast = super().hoist_shared(ast, args)

        # Declare the storage the block stages its columns in
        self.decls = list(self.columns.values())

        return ast

    def lower(self, ast, env):
        ast = super().lower(ast, env)

        # Have the lanes of the block fill the storage of its columns
        if self.decls:
            lx, ly = (Var(lid) for lid in self.gen._lid)
            subs = {'cidx': lx, 'ridx': ly, 'rsz': Int(self.rows)}
            for d in self.decls:
                ixs = ''.join(unravel_index('_i', d.cdims))
                code = f'''
                    for (int _i = @ridx; _x < _nx && _i < {prod(d.cdims)};
                         _i += @rsz)
                        {d.name}{ixs}[@cidx] = {d.load[0].name}{ixs};'''
                stmt = parse_program(code).body[0]
                self.prologue.append(rename_vars(stmt, subs))
            self.prologue.append(ExprStmt(DSLCall('sync', [])))

        return ast

    def _read(self, decl, ixs):
        # Read the copy of the storage belonging to the column of the lane
        lx = Var(self.gen._lid[0])

        return Index(reduce(Index, ixs, Var(decl.name)), lx)

    def extents(self, decl):
        # Give each column of the block its own copy of the storage
        return f'{decl.cdimstr}[{self.cols}]'

    def check_storage(self, ast):
        # Count the bytes of storage staged for the columns of a block
        nbytes = sum(d.nbytes(self.gen) for d in self.decls)*self.cols

        if nbytes > self.gen.backend.shared_budget:
            raise ValueError('Shared declarations exceed the block storage')

        return ast


class LaneGroup(BaseLaneModel):
    coop = True

    def __init__(self, gen, spec):
        super().__init__(gen)

        n = None if spec == 'fit' else int(spec)
        if n is not None and n < 1:
            raise ValueError(f'Invalid coop specification {spec!r}')
        if gen.ndim != 1:
            raise ValueError('Cooperative kernels must be 1D')

        # Leave the lanes and groups of a fitted block until storage is known
        if n is None:
            self.size = self.groups = None
        # Round a block of one group up to whole subgroups and cap its lanes
        else:
            n = 1 << (n - 1).bit_length()
            self.size = min(gen.max_lanes, max(gen.backend.subgrpsz, n))
            self.groups = 1

        # Scratch slots through which the lanes exchange reduced values
        self.scratch = ScratchSlots()

        # Whether the lanes write an argument in place of shared storage
        self.stored = False

    def _slots(self, ast):
        decls = [d for d in walk_ast(ast) if isinstance(d, VarDecl)]
        types, slots = {d.name: d.vtype for d in decls}, ScratchSlots()

        # Reserve the slots through which the lanes exchange values at once
        for node in walk_ast(ast):
            match node:
                case Region('allreduce', _, names):
                    slots.reserve([types[v] for v in names if v in types])

        return slots

    def _fit_groups(self, ast):
        gen, slots = self.gen, self._slots(ast)
        budget, n = gen.backend.shared_budget, gen.max_lanes

        # Start from a SoA group of items, one per subgroup of the block
        sgsz = gen.backend.subgrpsz
        g = min(1 << (gen.soasz.bit_length() - 1), max(n // sgsz, 1))

        # Halve the groups until their copies of the storage fit the budget
        while g > 1 and g*self.storage_bytes(slots, n // g) > budget:
            g //= 2

        return g

    def hoist_shared(self, ast, args):
        self._outargs = {n for n, a in args.items() if a.intent != 'in'}
        self.args = args

        ast, self.decls = self._split_shared(ast, args)
        self.shared = {d.name: d for d in self.decls}
        self._check_stores(ast)
        self._check_sync(ast)

        # Settle the lane-groups of a block fitted to the shared budget
        if self.groups is None:
            self._check_barriers(ast)
            self.groups = self._fit_groups(ast)

        # Divide the lanes of the block among its groups
        if self.size is None:
            self.size = self.gen.max_lanes // self.groups

        return ast

    @property
    def indexing(self):
        # Choose how the lanes of a block find their lane-group and its item
        cls = GroupsIndexing if self.groups > 1 else GroupIndexing

        return cls(self.gen, self.size, self.groups)

    def _check_barriers(self, ast):
        conds = [n for n in walk_ast(ast) if isinstance(n, If)]
        nodes = [n for c in conds for n in walk_ast(c)]

        # Check every lane of a fitted block reaches its barriers
        for n in nodes:
            match n:
                case DSLCall('sync', _) | Region('allreduce', _, _):
                    raise ValueError('Barriers of a fitted block can not be '
                                     'conditional')

    def _check_stores(self, ast):
        # Check the stores are top-level statements
        for stmt in body_scope(ast).body:
            call = stmt.expr if isinstance(stmt, ExprStmt) else None

            for n in walk_ast(stmt):
                match n:
                    case DSLCall('store', _) if n is not call:
                        raise ValueError('Stores must be top-level statements')

    def _check_sync(self, ast):
        pending, dirty, storing = set(), set(), set()

        # Check loaded storage is used after a barrier
        for stmt in body_scope(ast).body:
            match stmt:
                case ExprStmt(DSLCall('load', [Var(name), _])):
                    pending.add(name)
                    continue
                case ExprStmt(DSLCall('sync', [])):
                    pending, dirty, storing = set(), set(), set()
                    continue
                case ExprStmt(DSLCall('store', [Var(aname), expr])):
                    names = array_names(expr)

                    # Check stored storage is written before a barrier
                    for name in names:
                        if name in dirty:
                            raise ValueError(f'Shared {name!r} is stored '
                                             'before a top-level sync')
                    storing.add(aname)
                    storing.update(n for n in names if n in self.shared)

            for n in walk_ast(stmt):
                match n:
                    case Var(name) if name in pending:
                        raise ValueError(f'Shared {name!r} is used before '
                                         'a top-level sync')
                    case _ if name := _written(n):
                        if name in storing:
                            isarg = name in self.args
                            kind = 'Argument' if isarg else 'Shared'
                            raise ValueError(f'{kind} {name!r} is written '
                                             'while the lanes store it')
                        if name in self.shared:
                            dirty.add(name)

    def copy_stmt(self, op, clamp, comments):
        idx, ng, step = self.indexing, self.groups, self.size*self.groups

        # Check each array of the expression broadcasts over the extents copied
        for name, (sdims, _) in op.srcs.items():
            if sdims and list(sdims) != list(op.cdims[-len(sdims):]):
                raise ValueError(f'{name!r} does not broadcast over '
                                 f'{op.name!r}')

        # Copy a single bf16 array into bf16 as it stands
        isbf16 = isinstance(op.expr, Var) and op.dtype == 'bf16'
        plain = isbf16 and op.srcs[op.expr.name][1] == 'bf16'

        # Count the entries walked by the block loop and unravel their indices
        n, ixs = prod(op.cdims), unravel_index(idx.entry, op.cdims)

        def entry(name, cdims, dtype=None):
            # Index the array by the trailing indices its extents cover
            grp = idx.group if name in self.shared else ''
            src = name + grp + ''.join(ixs[len(ixs) - len(cdims):])

            # Convert the entries of a bf16 array unless they are copied
            if dtype == 'bf16' and not plain:
                return f'bf16_to_f32({src})'
            else:
                return src

        # Convert the result for a bf16 target unless it is copied
        if op.dtype == 'bf16' and not plain:
            copy = f'{entry(op.name, op.cdims)} = f32_to_bf16(@cpexpr);'
        else:
            copy = f'{entry(op.name, op.cdims)} = @cpexpr;'

        # Have each lane copy its entries in turn
        code = f'''
            for (int _li = {idx.start}; _li < {n*ng}; _li += {step})
            {{
                {idx.ritem if clamp else idx.witem}
                {copy}
            }}'''
        stmt = replace(parse_program(code).body[0], comments=comments)

        # Index the arrays of the expression by the entry of the loop
        entries = {k: parse_expr(entry(k, *v)) for k, v in op.srcs.items()}
        expr = rename_vars(op.expr, entries)

        # Place the expression at the marker and bind the lane intrinsics
        return rename_vars(stmt, self._lane_subs() | {'cpexpr': expr})

    def stores_arg(self, expr):
        return _written(expr) in self._outargs

    def _named(self, stmts, name):
        rest = [s for s in stmts if not _copied(s)]
        nodes = [n for s in rest for n in walk_ast(s)]

        # Check if a statement other than a load or store names the array
        return any(isinstance(n, Var) and n.name == name for n in nodes)

    def _entries(self, stmts, name, sub):
        ndim = len(self.shared[name].cdims)

        def entry(node):
            aname, ixs = unwrap_index(node)

            # Replace the fully indexed entries of the storage
            if _copied(node):
                return node
            elif aname == name and len(ixs) == ndim:
                return sub([entry(i) for i in ixs])
            else:
                return node.map(entry)

        return [entry(s) for s in stmts]

    def _store_inplace(self, body, d):
        ios = {i: io for i, s in enumerate(body) if (io := _copied(s))}
        lds = [i for i, io in ios.items() if io[:2] == ('load', d.name)]
        sts = [i for i, (kind, _, e) in ios.items()
               if kind == 'store' and d.name in array_names(e)]

        # Check for a single store of the storage
        if len(sts) != 1:
            return None

        # Check it combines the storage with arguments it does not write
        _, aname, sexpr = ios[si := sts[0]]
        arg, held = self.args.get(aname), {*self.shared, aname}
        if arg is None or held & set(array_names(sexpr)) != {d.name}:
            return None

        # Check the argument has the extents and type of the storage
        if arg.cdims != d.cdims or arg.dtype != d.dtype:
            return None

        # Check the rest of the body leaves the argument alone
        rest = [io for i, io in ios.items() if i not in (si, *lds)]
        shares = any(t == aname or aname in array_names(e) for _, t, e in rest)
        late = self._named(body[si + 1:], d.name)
        if shares or late or self._named(body, aname):
            return None

        def sub(ixs):
            return reduce(Index, ixs, Var(aname))

        # Have the body write the entries of the argument
        new = self._entries(body, d.name, sub)
        if self._named(new, d.name):
            return None

        # Load into the argument and store to it what differs from it
        for i in (*lds, si):
            expr = rename_vars(ios[i][2], {d.name: Var(aname)})
            if expr == Var(aname):
                new[i] = replace(new[i], expr=None)
            else:
                call = DSLCall('store', [Var(aname), expr])
                new[i] = replace(new[i], expr=call)

        return new

    def _load_inplace(self, body, d):
        ios = {i: io for i, s in enumerate(body) if (io := _copied(s))}
        lds = [i for i, io in ios.items() if io[:2] == ('load', d.name)]
        stores = [io for io in ios.values() if io[0] == 'store']
        nodes = [n for s in body if not _copied(s) for n in walk_ast(s)]

        # Collect the arrays the body or a store writes and those a store reads
        written = {_written(n) for n in nodes} | {t for _, t, _ in stores}
        stored = {n for _, _, e in stores for n in array_names(e)}

        # Check the storage is loaded once and neither written nor stored
        if len(lds) != 1 or d.name in written | stored:
            return None

        # Check the arguments it loads go unwritten
        if written & set(d.srcs):
            return None

        # Check bf16 storage is a copy of a bf16 argument
        types = {t for _, t in d.srcs.values()}
        plain = isinstance(d.expr, Var) and types == {d.dtype}
        if d.dtype == 'bf16' and not plain:
            return None

        def sub(ixs):
            ents = {}

            # Index each argument by the trailing indices its extents cover
            for n, (dims, dtype) in d.srcs.items():
                e = reduce(Index, ixs[len(ixs) - len(dims):], Var(n))
                if dtype == 'bf16' and not plain:
                    ents[n] = Call(Var('bf16_to_f32'), [e])
                else:
                    ents[n] = e

            # Convert the expression to the type of the storage
            expr = rename_vars(d.expr, ents)
            return expr if types == {d.dtype} else Cast(d.dtype, expr)

        # Have the body read the entries of the arguments
        new = self._entries(body, d.name, sub)
        if self._named(new, d.name):
            return None

        new[lds[0]] = replace(new[lds[0]], expr=None)
        return new

    def _inplace(self, ast, loads):
        body = body_scope(ast).body

        # Have the lanes work on the arguments in place of their storage
        for d in list(self.decls):
            stored = self._store_inplace(body, d)
            if stored or not loads:
                new = stored
            else:
                new = self._load_inplace(body, d)

            if new:
                body = new
                self.decls.remove(d)
                del self.shared[d.name]

            # Note if the lanes write an argument in place of the storage
            self.stored |= bool(stored)

        return with_body_scope(ast, body)

    def _fits(self, ast):
        nbytes = self.storage_bytes(self._slots(ast), self.size)

        # Check the copies held by the groups of a block fit its storage
        return nbytes*self.groups <= self.gen.backend.shared_budget

    def lower(self, ast, env):
        subs = self._lane_subs()

        # Work in place for a lone lane
        if self.size*self.groups == 1:
            ast = self._inplace(ast, True)
        # Write in place first for a block which can not hold its copies
        elif not self._fits(ast):
            ast = self._inplace(ast, False)

            # Read in place as well where the copies it loads still do not fit
            if not self._fits(ast):
                ast = self._inplace(ast, True)

        names = {d.name for d in self.decls}
        ast = self.indexing.index_by_group(ast, names)
        ast = LaneLowering(self, subs, env).visit(ast)

        # Declare the scratch slots alongside the shared storage
        self.decls += [SharedDecl(d, {})
                       for d in self.scratch.decls(self.size)]

        return ast

    def _reduce_types(self, names, env):
        # Check the reduced variables are declared scalars
        types = []
        for v in names:
            match env.lookup_decl(v):
                case (t, []):
                    types.append(t)
                case None:
                    raise ValueError(f'Undeclared reduce variable {v!r}')
                case _:
                    raise ValueError(f'Reduce variable {v!r} must be a scalar')

        return types

    def _partner_reads(self, names, combine):
        def rewrite(node):
            match node:
                case DSLCall('partner', [Var(v)]) if v in names:
                    return Var(f'_p_{v}')
                case DSLCall('partner', [Var(v)]):
                    raise ValueError(f'Allreduce does not exchange {v!r}')
                case DSLCall('partner', _):
                    raise ValueError('@partner takes a reduced variable')
                case _:
                    return node.map(rewrite)

        # Replace each @partner(v) in the combine step by its exchanged value
        return [rewrite(s) for s in combine]

    def allreduce(self, names, combine, env, comments):
        types = self._reduce_types(names, env)
        combine = self._partner_reads(names, combine)

        # Skip the exchange for a group of one lane
        if self.size == 1:
            return ExprStmt(None, comments=comments)

        # Exchange the values across the group with a butterfly
        slots = self.scratch.reserve(types)
        code = butterfly(names, types, slots, self.size, combine)
        blk = Block(code, comments=comments)

        return self.indexing.index_by_group(blk, set(self.scratch.names))

    def _lane_subs(self):
        # Bind @lidx and @lsz to the index of a lane and the size of its group
        if self.size == 1:
            lidx = Int(0)
        else:
            lidx = self.indexing.lidx()

        return {'lidx': lidx, 'lsz': Int(self.size)}

    def sync_stmt(self, comments):
        # Emit a barrier of the lanes, or nothing for a block of one lane
        if self.size == 1 and self.groups == 1:
            return ExprStmt(None, comments=comments)
        else:
            return ExprStmt(DSLCall('sync', []), comments=comments)

    def extents(self, decl):
        # Give each lane-group of the block its own copy of the storage
        return f'{self.indexing.extent}{decl.cdimstr}'

    def validate_args(self, vargs):
        if any(v.isreduce for v in vargs):
            raise ValueError('Coop kernels do not support reduce arguments')

    def storage_bytes(self, slots, size):
        # Count the bytes of shared storage and exchange slots of a lane-group
        decls = [SharedDecl(d, {}) for d in slots.decls(size)]

        return sum(d.nbytes(self.gen) for d in [*self.shared.values(), *decls])

    def check_storage(self, ast):
        gen, ng = self.gen, self.groups
        nbytes = self.storage_bytes(self.scratch, self.size)

        # Check the copies held by the groups of a block fit its storage
        if nbytes*ng > gen.backend.shared_budget:
            raise ValueError('Shared declarations exceed the block storage')

        # Fence the barriers of lanes which write a broadcast argument
        written = [a for a in gen.vectargs if a.isbroadcast]
        bcast = any(a.intent != 'in' for a in written)

        # Fence those of lanes which write an argument in place of storage
        self.fenced = bcast or (self.stored and self.size*ng > 1)

        def resync(node):
            match node:
                case DSLCall('sync', []):
                    return DSLCall('sync_gmem', [])
                case _:
                    return node.map(resync)

        # Have the barriers also order the accesses to global memory
        return resync(ast) if self.fenced and self.size*ng > 1 else ast
