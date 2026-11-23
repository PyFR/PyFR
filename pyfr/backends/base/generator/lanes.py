from dataclasses import replace
from functools import reduce
import itertools as it
from math import prod

from pyfr.backends.base.generator.reduce import ScratchSlots, butterfly
from pyfr.dsl.nodes import (Assign, Block, Cast, Comment, DSLCall, DSLVar,
                            ExprStmt, Index, Int, MultiDecl, Postfix, Region,
                            Transformer, Unary, Var, VarDecl, unwrap_index,
                            walk_ast)
from pyfr.dsl.rewriter import parse_program, rename_vars


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
            case (DSLCall('sync' | 'load' as r, _) |
                  Region('allreduce' as r, _, _)):
                raise ValueError(f'{r} requires a coop kernel')


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


class SharedDecl:
    def __init__(self, decl, args):
        self.name = decl.name
        self.dtype = decl.vtype

        # Argument whose entries are loaded into the storage, if any
        self.load = None

        match decl.init:
            case DSLCall('load', [Var(aname)]):
                self._init_load(decl, args.get(aname), aname)
            case DSLCall('load', _):
                raise ValueError(f'Shared {self.name!r} must load from an '
                                 'argument')
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

    def _init_load(self, decl, arg, aname):
        # Validation
        if decl.sizes:
            raise ValueError(f'Shared {self.name!r} takes its extents from '
                             'the argument it loads')
        if arg is None:
            raise ValueError(f'Shared {self.name!r} loads from an unknown '
                             f'argument {aname!r}')
        if arg.isview or arg.ismpi or arg.isscalar or arg.isreduce:
            raise ValueError(f'Argument {aname!r} can not be loaded')
        if arg.intent == 'out':
            raise ValueError(f'Loaded argument {aname!r} must be of intent '
                             'in or inout')
        if not arg.cdims:
            raise ValueError(f'Argument {aname!r} is not an array')

        # Take the shape of the argument and, unless given, its type
        self.load = arg
        self.dtype = decl.vtype or arg.dtype
        self.cdims = list(arg.cdims)


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
                decl = next(d for d in model.decls if d.name == name)
                return self.visit(model.load_stmt(decl, cs))
            case DSLCall('load', _):
                raise ValueError('Loads must declare shared storage')
            case Region('allreduce', body, names, comments=cs):
                red = model.allreduce(list(names), body, self.env, cs)
                return self.visit(red)
            case DSLCall('partner', _):
                raise ValueError('@partner must be inside an allreduce region')
            case _:
                return super().visit(node)


class BaseLaneModel:
    coop = False

    def __init__(self, gen):
        self.gen = gen
        self.size = 1
        self.decls = []
        self.prologue = []
        self.staged = set()

    def _split_shared(self, ast, args):
        body, decls = [], []

        # Separate the top-level shared declarations from the body
        for stmt in body_scope(ast).body:
            if new := [SharedDecl(d, args) for d in shared_decls(stmt)]:
                decls += new
            else:
                body.append(stmt)

            # Leave a load of the storage in place of its declaration
            for d in new:
                if d.load is not None:
                    load = DSLCall('load', [Var(d.name), Var(d.load.name)])
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
        self.staged = {d.load.name for d in decls}
        self.columns = {d.name: d for d in decls}

        # Drop their loads, which the block carries out before its rows
        return with_body_scope(ast, body_scope(ast).body[len(decls):])

    def _check_staged(self, decl):
        if decl.load is None:
            raise ValueError(f'Shared {decl.name!r} needs a column argument')

        # Check a column argument of the type of the storage is loaded
        arg = decl.load
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
        return reduce(Index, ixs, Var(decl.load.name))


class StagedRowLanes(RowLanes):
    def __init__(self, gen):
        super().__init__(gen)

        # Items of a block, one per column, and the lanes splitting their rows
        self.cols, self.rows = gen.block2d[:2]

    def hoist_shared(self, ast, args):
        ast = super().hoist_shared(ast, args)

        # Declare the storage the block stages its columns in
        self.decls = list(self.columns.values())
        self._check_storage()

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
                        {d.name}{ixs}[@cidx] = {d.load.name}{ixs};'''
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

    def _check_storage(self):
        # Count the bytes of storage staged for the columns of a block
        nbytes = sum(d.nbytes(self.gen) for d in self.decls)*self.cols

        if nbytes > self.gen.backend.shared_budget:
            raise ValueError('Shared declarations exceed the block storage')


class LaneGroup(BaseLaneModel):
    coop = True

    def __init__(self, gen, spec):
        super().__init__(gen)

        n = int(spec)
        if n < 1:
            raise ValueError(f'Invalid coop specification {spec!r}')
        if gen.ndim != 1:
            raise ValueError('Cooperative kernels must be 1D')

        # Round the lanes requested up to whole subgroups and cap them
        n = 1 << (n - 1).bit_length()
        self.size = min(gen.max_lanes, max(gen.backend.subgrpsz, n))

        # Scratch slots through which the lanes exchange reduced values
        self.scratch = ScratchSlots()

    def hoist_shared(self, ast, args):
        ast, self.decls = self._split_shared(ast, args)
        self._check_loads(ast)
        self._check_storage()

        return ast

    def _check_loads(self, ast):
        pending = set()

        # Check loaded storage is used after a barrier
        for stmt in body_scope(ast).body:
            match stmt:
                case ExprStmt(DSLCall('load', [Var(name), _])):
                    pending.add(name)
                    continue
                case ExprStmt(DSLCall('sync', [])):
                    pending.clear()
                    continue

            for n in walk_ast(stmt):
                match n:
                    case Var(name) if name in pending:
                        raise ValueError(f'Shared {name!r} is used before '
                                         'a top-level sync')

    def load_stmt(self, decl, comments):
        arg, n = decl.load, prod(decl.cdims)
        ixs = ''.join(unravel_index('_li', decl.cdims))

        # Convert the entries of a bf16 argument to the type of the storage
        if arg.dtype == 'bf16' and decl.dtype != 'bf16':
            src = f'bf16_to_f32({arg.name}{ixs})'
        else:
            src = f'{arg.name}{ixs}'

        # Have the lanes of the group copy the entries of the argument
        code = f'''
            for (int _li = @lidx; _li < {n}; _li += @lsz)
                {decl.name}{ixs} = {src};'''
        stmt = parse_program(code).body[0]

        return replace(stmt, comments=comments)

    def lower(self, ast, env):
        subs = self._lane_subs()
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
                case (t, None):
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

        return Block(code, comments=comments)

    def _lane_subs(self):
        # Bind @lidx and @lsz to the index of a lane and the size of its group
        if self.size == 1:
            lidx = Int(0)
        else:
            lidx = Cast('int', Var(self.gen._lid[0]))

        return {'lidx': lidx, 'lsz': Int(self.size)}

    def sync_stmt(self, comments):
        # Emit a barrier of the lanes, or nothing for a group of one lane
        if self.size == 1:
            return ExprStmt(None, comments=comments)
        else:
            return ExprStmt(DSLCall('sync', []), comments=comments)

    def validate_args(self, vargs):
        if any(v.isreduce for v in vargs):
            raise ValueError('Coop kernels do not support reduce arguments')

    def _check_storage(self):
        # Count the bytes of shared storage declared by the body
        nbytes = sum(d.nbytes(self.gen) for d in self.decls)

        if nbytes > self.gen.backend.shared_budget:
            raise ValueError('Shared declarations exceed the lane-group '
                             'storage')
