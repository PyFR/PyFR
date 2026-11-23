from dataclasses import replace

from pyfr.dsl.nodes import (ArrayInit, Assign, Binary, Block, Comment,
                            DoWhile, ExprStmt, For, If, Index, Int,
                            MultiDecl, Number, Postfix, Program, Region,
                            Transformer, Unary, Var, VarDecl, While,
                            unwrap_index, walk_ast)


class NotVectorisable(Exception):
    pass


class SIMDLowering(Transformer):
    # Loop variable and count of the lanes of a SoA group
    lane = '_xj'
    nlanes = '_xjn'

    def __init__(self, soasz, shared):
        self.soasz = soasz

        # Storage shared by the lanes of an item, which differs between items
        self.shared = set(shared)

    def visit(self, node):
        match node:
            case Program(body) | Block(body) if self._straight(body):
                decls = [n for n in walk_ast(node) if isinstance(n, VarDecl)]

                # Walk the block a lane at a time, each holding its locals
                loop = self._lane_loop(body, {d.name for d in decls})

                return replace(node, body=[loop])
            case _:
                return super().visit(node)

    def lower(self, ast):
        # Find the locals which take a different value in each lane
        self.locals = {n.name for n in walk_ast(ast) if isinstance(n, VarDecl)}
        self.varying = set(self.shared)
        self._find_varying(ast)

        # Check control flow is the same in every lane
        self._check_uniform(ast)

        return self.visit(ast)

    def rewrite(self, node):
        # Enclose the lane-valued statements of each body in lane loops
        match node:
            case Program() | Block() | Region():
                return replace(node, body=self._block(node.body))
            case _:
                return node

    def _visit_stmt(self, node):
        new = self.visit(node)

        # Give a lone lane-valued statement a lane loop of its own
        if isinstance(new, ExprStmt) and self._is_varying(new):
            return Block([self._lane_loop([new])])
        else:
            return new

    def _straight(self, stmts):
        stmts = [s for s in stmts if not isinstance(s, Comment)]

        # Check if every statement of a block is lane-valued
        return bool(stmts) and all(map(self._laned, stmts))

    def _laned(self, stmt):
        match stmt:
            case VarDecl(name=name):
                return name in self.varying
            case MultiDecl(decls):
                return all(d.name in self.varying for d in decls)
            case ExprStmt():
                return self._is_varying(stmt)
            case Block(body):
                return self._straight(body)
            case _:
                return False

    def _is_varying(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]

        names = self.varying | {self.lane}
        return any(isinstance(n, Var) and n.name in names
                   for node in nodes for n in walk_ast(node))

    def _find_varying(self, ast):
        # Iterate as a store of a lane value makes its target lane-valued
        nvarying = -1
        while len(self.varying) != nvarying:
            nvarying = len(self.varying)

            for node in walk_ast(ast):
                match node:
                    case VarDecl(name=name, init=init) if init is not None:
                        if self._is_varying(init):
                            self.varying.add(name)
                    case Assign(left, right):
                        name, ix = unwrap_index(left)
                        srcs = [right, *ix]
                        if name in self.locals and self._is_varying(srcs):
                            self.varying.add(name)
                    case Postfix(_, opnd) | Unary('++' | '--', opnd):
                        name, ix = unwrap_index(opnd)
                        if name in self.locals and self._is_varying(ix):
                            self.varying.add(name)

    def _check_uniform(self, ast):
        for node in walk_ast(ast):
            match node:
                case If(cond=c) | While(cond=c) | DoWhile(cond=c):
                    if self._is_varying(c):
                        raise NotVectorisable('Lane-dependent condition')
                case For(init=i, cond=c, step=s):
                    parts = [x for x in (i, c, s) if x is not None]
                    if self._is_varying(parts):
                        raise NotVectorisable('Lane-dependent loop')

    def _block(self, stmts):
        out, lanes, group, comments = [], [], [], []

        def enclose(*s):
            # Gather consecutive lane-valued statements for a single lane loop
            group.extend([*comments, *s])
            comments.clear()

        def emit(s):
            if group:
                out.append(self._lane_loop(group))
                group.clear()

            out.extend([*comments, s])
            comments.clear()

        for s in stmts:
            match s:
                case VarDecl() | MultiDecl():
                    decls = s.decls if isinstance(s, MultiDecl) else [s]
                    for d in decls:
                        if d.name in self.varying:
                            lanes.append(self._lane_decl(d))
                            enclose(*self._lane_init(d))
                        else:
                            emit(d)
                case Comment():
                    comments.append(s)
                case ExprStmt() if self._is_varying(s):
                    enclose(s)
                case _:
                    emit(s)

        if group:
            out.append(self._lane_loop(group))

        # Declare the lane-valued locals as arrays at the top of the block
        return [*lanes, *out, *comments]

    def _lane_decl(self, d):
        sizes = [*d.sizes, Int(self.soasz)]

        # Extend zero initialisers to the lane dimension unchanged
        if isinstance(d.init, ArrayInit) and self._is_zero(d.init):
            return VarDecl(d.vtype, d.name, sizes, d.init)
        else:
            return VarDecl(d.vtype, d.name, sizes)

    def _is_zero(self, init):
        if isinstance(init, ArrayInit):
            return all(self._is_zero(v) for v in init.values)
        else:
            return isinstance(init, Number) and init.value == 0

    def _lane_init(self, d):
        def assigns(target, init):
            if isinstance(init, ArrayInit):
                for i, v in enumerate(init.values):
                    yield from assigns(Index(target, Int(i)), v)
            else:
                yield ExprStmt(Assign(target, init))

        if d.init is None:
            return []
        elif isinstance(d.init, ArrayInit) and self._is_zero(d.init):
            return []
        # Assign the initialiser of a lane-valued local entry by entry
        else:
            return list(assigns(Var(d.name), d.init))

    def _lane_loop(self, stmts, held=frozenset()):
        xj = Var(self.lane)

        # Index the lane-valued names other than those each lane holds
        indexed = self.varying - held

        def lane(node):
            match node:
                # v -> v[_xj]
                case Var(name) if name in indexed:
                    return Index(node, xj)
                # v[i][k] -> v[i][k][_xj], with i and k transformed likewise
                case Index() if unwrap_index(node)[0] in indexed:
                    return Index(indices(node), xj)
                # Otherwise transform the operands of the node
                case _:
                    return node.map(lane)

        def indices(node):
            # Rebuild an index chain, leaving its root array untouched
            if isinstance(node.array, Index):
                return Index(indices(node.array), lane(node.index))
            else:
                return Index(node.array, lane(node.index))

        body = Block([lane(s) for s in stmts])
        init = VarDecl('int', self.lane, init=Int(0))
        cond = Binary('<', xj, Var(self.nlanes))
        loop = For(init, cond, Postfix('++', xj), body)

        return Region('simd', [loop])
