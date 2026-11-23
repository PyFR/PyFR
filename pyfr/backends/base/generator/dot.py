from dataclasses import replace
import functools as ft
import itertools as it
from math import prod

from pyfr.backends.base.generator.lanes import _written
from pyfr.dsl.nodes import (ArrayInit, Assign, Binary, Block, Break, Call,
                            DoWhile, DSLCall, DSLVar, ExprStmt, For, Index,
                            Int, MultiDecl, Region, Transformer, Unary, Var,
                            VarDecl, While, walk_ast)


def _for(k, stop, step, body):
    init = VarDecl('int', k.name, init=Int(0))
    cond = Binary('<', k, Int(stop))
    step = Assign(k, Binary('+', k, Int(step)))

    return For(init, cond, step, Block(body))


class DotOperands:
    def __init__(self, operand, decl, dargs):
        w, m, r, v = self._match_args(dargs)
        shape, nw, nrhs = self._match_extents(decl)

        # Check the result is not accumulated in an encoded type
        if decl.vtype == 'bf16':
            raise ValueError(f'Dot product {decl.name!r} cannot be bf16')

        # Check the matrix is two dimensional and the vector spans its columns
        mtype, mdims, tile = operand(m)
        vtype, vdims, _ = operand(v)
        if len(mdims) != 2:
            raise ValueError(f'Matrix {m!r} of a dot product must be two '
                             'dimensional')
        if prod(vdims) < mdims[1]*nrhs:
            raise ValueError(f'Vector {v!r} is too short for {m!r}')

        # Check each row of the result has a row of weights over the columns
        if w is None and len(shape) == 2:
            raise ValueError(f'Dot product {decl.name!r} needs weights for '
                             'the rows of its result')
        elif w is None:
            wgt = wtype = None
        else:
            wgt = Var(w)
            wtype, wdims, _ = operand(w)
            if not wdims or wdims[:-1] != shape[:-1] or wdims[-1] < mdims[1]:
                raise ValueError(f'Weights {w!r} do not match the result '
                                 f'and columns of {m!r}')

        # Accumulate from zero in the declared type
        zero = ArrayInit([Int(0)]) if shape else Int(0)
        self.decl, self.ttype = replace(decl, init=zero), decl.vtype
        self.shape, self.nw, self.nrhs = shape, nw, nrhs

        # Row of the matrix, its columns and the shape of any tiles
        self.row, self.mtype = Index(Var(m), r), mtype
        self.ncol, self.tile = mdims[1], tile

        # Vector and weights of the product
        self.vec, self.vtype, self.vdims = Var(v), vtype, vdims
        self.wgt, self.wtype = wgt, wtype

        # Collect the names of everything the product reads
        nodes = [n for a in dargs for n in walk_ast(a)]
        self.reads = {n.name for n in nodes if isinstance(n, Var)}

    @staticmethod
    def _match_args(dargs):
        match dargs:
            # Match @dot(w*m[r], v)
            case [Binary('*', Var(w), Index(Var(m), r)), Var(v)]:
                return w, m, r, v
            # Match @dot(m[r], v)
            case [Index(Var(m), r), Var(v)]:
                return None, m, r, v
            case _:
                raise ValueError('Dot products take a matrix row and a vector')

    @staticmethod
    def _match_extents(decl):
        # Take the weight rows and vectors of the result from its extents
        match decl.sizes:
            # Match T a = @dot(...)
            case []:
                return [], 1, 1
            # Match T a[nrhs] = @dot(...)
            case [Int(nrhs)]:
                return [nrhs], 1, nrhs
            # Match T a[nw][nrhs] = @dot(...)
            case [Int(nw), Int(nrhs)]:
                return [nw, nrhs], nw, nrhs
            case _:
                raise ValueError(f'Dot product {decl.name!r} needs constant '
                                 'extents of at most two dimensions')

    def accumulate(self, target, k, entry):
        ttype, wgt = self.ttype, self.wgt

        # Hold the entry once for every weighting and vector
        e, entry = Var('_de'), self._decode(entry, self.mtype)
        stmts = [VarDecl(ttype, e.name, init=entry)]
        for kw in range(self.nw):
            # Weight the entry once for the vectors of this row
            if wgt is None:
                ew = e
            else:
                wrow = Index(wgt, Int(kw)) if len(self.shape) == 2 else wgt
                we = Binary('*', self._decode(Index(wrow, k), self.wtype), e)
                ew = Var(f'_dw{kw}')
                stmts.append(VarDecl(ttype, ew.name, init=we))

            # Accumulate the weighted entry into each result
            for j in range(self.nrhs):
                ix = [Int(kw), Int(j)][2 - len(self.shape):]
                tj = ft.reduce(Index, ix, target)
                vkj = self._decode(self._ventry(k, j), self.vtype)
                term = Binary('*', ew, vkj)
                stmts.append(ExprStmt(Assign(tj, Binary('+', tj, term))))

        return stmts

    @staticmethod
    def _decode(entry, dtype):
        # Convert an entry of an operand held in an encoded type
        if dtype == 'bf16':
            return Call(Var('bf16_to_f32'), [entry])
        else:
            return entry

    def _ventry(self, k, j):
        vec, vdims, nrhs = self.vec, self.vdims, self.nrhs

        # Index entry k of vector j in the row-major flattening of the vector
        if vdims[1:] == [nrhs]:
            return Index(Index(vec, k), Int(j))
        elif nrhs > 1:
            k = Binary('+', Binary('*', k, Int(nrhs)), Int(j))

        # Index entry k of the row-major flattening of the array
        ixs = [k]
        for d in map(Int, reversed(vdims[1:])):
            i = ixs[0]
            ixs = [Binary('/', i, d), Binary('%', i, d), *ixs[1:]]

        return ft.reduce(Index, ixs, vec)


class TileRow:
    k0 = Var('_dk0')

    def __init__(self, row, mtype, tcols):
        self.row, self.mtype, self.tcols = row, mtype, tcols

    def col(self, i):
        return Binary('+', self.k0, i)

    def entry(self, i):
        return Index(self.row, self.col(i))

    def load(self, cols):
        return cols


class LoadedTileRow(TileRow):
    def entry(self, i):
        return Index(Var('_dr'), i)

    def load(self, cols):
        dr = Var('_dr')

        # Load the row of a tile as one ahead of its columns
        ldecl = VarDecl(self.mtype, dr.name, [Int(self.tcols)])
        largs = [dr, Unary('&', Index(self.row, self.k0))]
        load = ExprStmt(Call(Var('PYFR_LOAD_VEC'), largs))

        return [Block([ldecl, load, *cols])]


class TiledProduct:
    def __init__(self, ops, gen):
        self.ops, self.decl, self.target = ops, ops.decl, Var(ops.decl.name)

        # Load the row of a tile at once when it is as wide as a lane's load
        tbytes = ops.tile[1]*gen.npdtype(ops.mtype).itemsize
        tilerow = LoadedTileRow if tbytes == gen.lane_load_bytes else TileRow
        self.tcols = ops.tile[1]
        self.tilerow = tilerow(ops.row, ops.mtype, self.tcols)

        # Count the tiles a lane keeps in flight
        self.tiles_inflight = max(1, gen.bytes_inflight // tbytes)

    def lower(self):
        # Walk the columns of the row tile by tile
        return [self.decl, *self._tiles()]

    def _cols(self, n):
        i, tilerow = Var('_di'), self.tilerow
        k, entry = tilerow.col(i), tilerow.entry(i)

        # Walk the columns a tile holds of the row unrolled
        col = self.ops.accumulate(self.target, k, entry)

        return [Region('unroll', [_for(i, n, 1, col)], (str(n),))]

    def _tile(self, n):
        return self.tilerow.load(self._cols(n))

    def _tiles(self):
        k0, tcols, ncol = self.tilerow.k0, self.tcols, self.ops.ncol

        # Count the columns of the full tiles of the row and its partial tile
        ncol_f, ncol_p = ncol - ncol % tcols, ncol % tcols

        # Walk the full tiles, unrolled by the number of loads kept in flight
        stmts = []
        if ncol_f:
            loop = _for(k0, ncol_f, tcols, self._tile(tcols))
            unroll = (str(self.tiles_inflight),)
            stmts.append(Region('unroll', [loop], unroll))

        # Finish with the partial tile
        if ncol_p:
            decl = VarDecl('int', k0.name, init=Int(ncol_f))
            stmts.append(Block([decl, *self._tile(ncol_p)]))

        return stmts


class StripProduct(TiledProduct):
    def __init__(self, ops, gen, rdecl, ri, nrows):
        super().__init__(ops, gen)
        self.rdecl, self.ri, self.nrows = rdecl, ri, nrows

        # Declare the sums of each row of the strip
        sums, zero = Var(f'_ds_{ops.decl.name}'), ArrayInit([Int(0)])
        sizes = [Int(s) for s in (nrows, *ops.shape)]
        self.decl = replace(ops.decl, name=sums.name, sizes=sizes, init=zero)

        # Accumulate into the sums of the row of the strip being walked
        self.target = Index(sums, ri)

    def assign_result(self):
        decl = self.ops.decl
        ixs = it.product(*map(range, self.ops.shape))
        ixs = [[Int(i) for i in ix] for ix in ixs]
        res, sums = Var(decl.name), self.target

        # Assign the sums of the row to its result entry by entry
        lhs = [ft.reduce(Index, ix, res) for ix in ixs]
        rhs = [ft.reduce(Index, ix, sums) for ix in ixs]
        copy = map(Assign, lhs, rhs)

        return [replace(decl, init=None, comments=()), *map(ExprStmt, copy)]

    def _cols(self, n):
        cols, tilerow = [], self.tilerow

        # Walk the columns a tile holds of the row one after another
        for c in map(Int, range(n)):
            k, entry = tilerow.col(c), tilerow.entry(c)
            cols.append(Block(self.ops.accumulate(self.target, k, entry)))

        return [Block(cols)]

    def _tile(self, n):
        # Read the part of a tile each row of the strip takes
        tile = [self.rdecl, *super()._tile(n)]
        loop = _for(self.ri, self.nrows, 1, tile)

        return [Region('unroll', [loop], ('1',))]


class DotLowering(Transformer):
    def __init__(self, gen, args):
        self.gen = gen
        self.args = args

    def visit(self, node):
        # Take the rows of a loop the lanes share out a strip at a time
        match node:
            case For(VarDecl(init=DSLVar('lidx'))):
                block = self._strips(node)

                return super().visit(node) if block is None else block
            # Expand the declarations a dot product initialises
            case VarDecl(init=DSLCall('dot', dargs)):
                return self._expand(node, [self.visit(a) for a in dargs])
            # Split a multiple declaration any of whose declarations expanded
            case MultiDecl(decls):
                stmts = [s for d in decls for s in self._stmts(d)]

                return node if stmts == decls else stmts
            # Reject a dot product which did not initialise a declaration
            case DSLCall('dot'):
                raise ValueError('@dot must initialise a declaration')
            case _:
                return super().visit(node)

    def _stmts(self, node):
        out = self.visit(node)

        return out if isinstance(out, list) else [out]

    @staticmethod
    def _escapes(stmt):
        # Check if a statement can break out of the loop enclosing it
        match stmt:
            case Break():
                return True
            case For() | While() | DoWhile():
                return False
            case _:
                return any(map(DotLowering._escapes, stmt.children()))

    def _strip_dot(self, stmt):
        # Find the operands of a dot product over a row of a tiled argument
        match stmt:
            case VarDecl(init=DSLCall('dot', dargs)):
                ops = DotOperands(self.operand, stmt, dargs)

                return ops if ops.tile else None
            case _:
                return None

    def _strip_rows(self, loop):
        lm, v = self.gen.lanemodel, Var(loop.init.name)

        # Check for a lone lane stepping through the rows it is given
        lone = lm.coop and lm.size == 1 and lm.groups == 1
        stepped = loop.step == Assign(v, Binary('+', v, DSLVar('lsz')))

        # Find the constant number of rows the loop walks
        match loop.cond:
            case Binary('<', Var(name), Int(n)) if name == v.name:
                return n if lone and stepped else None
            case _:
                return None

    def _strip_dots(self, loop):
        v = loop.init.name
        body = loop.body.body if isinstance(loop.body, Block) else [loop.body]

        # Split off the dot products which open the body from the rest
        ops = list(it.takewhile(bool, map(self._strip_dot, body)))
        rest = body[len(ops):]

        # Check what the products read is named by nothing after them
        read = {n for p in ops for n in p.reads} - {v}
        nodes = [n for s in rest for n in walk_ast(s)]
        named = {n.name for n in nodes if isinstance(n, Var)}
        named |= {p.decl.name for p in ops}

        # Check the rest neither leaves the loop early nor moves its variable
        moved = any(_written(n) == v for n in nodes)
        escapes = any(map(self._escapes, rest))

        if ops and not (read & named or moved or escapes):
            return ops, rest
        else:
            return None

    def _strip(self, ops, rest, rdecl, ri, nrows):
        sps = [StripProduct(p, self.gen, rdecl, ri, nrows) for p in ops]

        # Sum the products of every row of the strip tile by tile
        walk = [s for sp in sps for s in sp.lower()]

        # Carry out the rest of the body row by row
        row = [rdecl, *[s for sp in sps for s in sp.assign_result()]]
        row += [s for st in rest for s in self._stmts(st)]

        return [*walk, _for(ri, nrows, 1, row)]

    def _strips(self, loop):
        # Check for a lone lane whose loop opens with tiled dot products
        if (n := self._strip_rows(loop)) is None:
            return None
        if (dots := self._strip_dots(loop)) is None:
            return None

        # Take the rows a strip of the tile rows of the first matrix at a time
        ops, rest = dots
        tr, r0, ri = ops[0].tile[0], Var('_dr0'), Var('_dri')
        rdecl = replace(loop.init, init=Binary('+', r0, ri))

        # Walk the full strips and finish with the partial strip
        stmts, nfull = [], n - n % tr
        if nfull:
            strip = self._strip(ops, rest, rdecl, ri, tr)
            stmts.append(_for(r0, nfull, tr, strip))
        if n % tr:
            decl = VarDecl('int', r0.name, init=Int(nfull))
            strip = self._strip(ops, rest, rdecl, ri, n % tr)
            stmts.append(Block([decl, *strip]))

        return Block(stmts)

    def operand(self, name):
        shared = self.gen.lanemodel.shared

        # Find the type, extents and tile shape of an argument or shared array
        if name in self.args:
            a = self.args[name]
            if a.ismulti or a.isscalar:
                raise ValueError(f'Dot products cannot take {name!r}')

            tile = self.gen.tiling(a)[:2] if a.istiled else None

            return a.dtype, list(a.cdims), tile
        elif name in shared:
            d = shared[name]

            return d.dtype, list(d.cdims), None
        else:
            raise ValueError(f'Dot product operand {name!r} must be an '
                             'argument or shared array')

    def _expand(self, decl, dargs):
        ops, k = DotOperands(self.operand, decl, dargs), Var('_dk')

        # Walk the row by tiles if its matrix is tiled, else by entries
        if ops.tile:
            return TiledProduct(ops, self.gen).lower()
        else:
            # Walk the columns of the row
            walk = ops.accumulate(Var(decl.name), k, Index(ops.row, k))

            return [ops.decl, _for(k, ops.ncol, 1, walk)]
