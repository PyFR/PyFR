from dataclasses import dataclass, field, fields, replace
import itertools as it

# Node number counter
_nids = it.count()


def node(cls):
    # Make the class an immutable record
    cls = dataclass(cls, slots=True, frozen=True)

    # Note the fields set at construction which hold a child or a list of them
    kinds = {object: False, list: True, list | None: True}
    fs = (f for f in fields(cls) if f.init and f.type in kinds)
    cls._child_fields = tuple((f.name, kinds[f.type]) for f in fs)

    return cls


@node
class Node:
    nid: int = field(default_factory=_nids.__next__, init=False,
                     compare=False, repr=False)

    def children(self):
        for name, islist in self._child_fields:
            v = getattr(self, name)
            if islist:
                yield from (c for c in v or () if isinstance(c, Node))
            elif isinstance(v, Node):
                yield v

    def map(self, t):
        # Rebuild the node with each child field passed through t
        new, changed = {}, False
        for name, islist in self._child_fields:
            v = getattr(self, name)
            if islist and v:
                new[name] = [t(c) if isinstance(c, Node) else c for c in v]
                changed |= any(a is not b for a, b in zip(new[name], v))
            elif isinstance(v, Node):
                new[name] = t(v)
                changed |= new[name] is not v

        # Leaves and nodes with no children are handed back untouched
        if not changed:
            return self
        # Otherwise copy the node with its new children
        else:
            return replace(self, **new)


@node
class Stmt(Node):
    comments: tuple = field(default=(), kw_only=True)


@node
class Expr(Node):
    pass


@node
class Program(Node):
    body: list


@node
class VarDecl(Stmt):
    vtype: str
    name: str
    sizes: list | None = None
    init: object = None
    const: bool = field(default=False, kw_only=True)
    shared: bool = field(default=False, kw_only=True)


@node
class Typedef(Stmt):
    old: str
    new: str


@node
class ArrayInit(Expr):
    values: list


@node
class MultiDecl(Stmt):
    decls: list


@node
class Block(Stmt):
    body: list


@node
class Region(Stmt):
    kind: str
    body: list
    args: tuple = ()


@node
class If(Stmt):
    cond: object
    then: object
    else_: object = None


@node
class While(Stmt):
    cond: object
    body: object


@node
class For(Stmt):
    init: object
    cond: object
    step: object
    body: object


@node
class DoWhile(Stmt):
    body: object
    cond: object


@node
class Break(Stmt):
    pass


@node
class Continue(Stmt):
    pass


@node
class ExprStmt(Stmt):
    expr: object


@node
class Comment(Stmt):
    kind: str
    text: str


@node
class Number(Expr):
    value: int | float
    suffix: str = ''
    ishex: bool = False


@node
class Int(Number):
    pass


@node
class Float(Number):
    pass


@node
class String(Expr):
    value: str


@node
class Var(Expr):
    name: str


@node
class Binary(Expr):
    op: str
    left: object
    right: object


@node
class Unary(Expr):
    op: str
    operand: object


@node
class Postfix(Expr):
    op: str
    operand: object


@node
class Ternary(Expr):
    cond: object
    true: object
    false: object


@node
class Assign(Expr):
    left: object
    right: object


@node
class Call(Expr):
    func: object
    args: list


@node
class Index(Expr):
    array: object
    index: object


@node
class Cast(Expr):
    ctype: str
    expr: object


@node
class DSLCall(Expr):
    name: str
    args: list


@node
class DSLVar(Expr):
    name: str


PRECEDENCE = {
    ',': 1, '=': 2, '+=': 2, '-=': 2, '*=': 2, '/=': 2, '%=': 2,
    '<<=': 2, '>>=': 2, '&=': 2, '|=': 2, '^=': 2, '?': 3,
    '||': 4, '&&': 5, '|': 6, '^': 7, '&': 8, '==': 9, '!=': 9,
    '<': 10, '<=': 10, '>': 10, '>=': 10, '<<': 11, '>>': 11,
    '+': 12, '-': 12, '*': 13, '/': 13, '%': 13,
    # Prefix operators bind below ** so -a**b is -(a**b) as in Python
    'unary': 14, '**': 15,
    '++': 20, '--': 20, '(': 20, '[': 20
}

BINARY_OPS = frozenset([
    '+', '-', '*', '/', '%', '==', '!=', '<', '<=', '>', '>=', '&&', '||',
    '<<', '>>', '&', '|', '^', '**'
])
RIGHT_ASSOC = frozenset(['**'])

COMPOUND_ASSIGN = frozenset([
    '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^='
])

UNARY_OPS = frozenset(['+', '-', '!', '~', '++', '--'])
POSTFIX_OPS = frozenset(['++', '--'])
PRECISE_REGIONS = frozenset(['fp-precise'])


def unwrap_index(expr):
    indices = []
    while isinstance(expr, Index):
        indices.append(expr.index)
        expr = expr.array

    # Indices accumulate innermost first, so hand them back reversed
    if isinstance(expr, Var):
        return expr.name, indices[::-1]
    else:
        return None, []


def walk_ast(node):
    yield node

    for c in node.children():
        yield from walk_ast(c)


def walk_ast_regions(node, kinds=frozenset()):
    yield node, kinds

    # Nodes beneath a region carry its kind in their context
    if isinstance(node, Region):
        kinds = kinds | {node.kind}

    for c in node.children():
        yield from walk_ast_regions(c, kinds)


class Transformer:
    # Scope of the declarations in effect, for passes which track them
    env = None

    def __init__(self, env=None):
        self.env = env

    def visit(self, node):
        # Rebuild the children of a node, then let the pass rewrite it
        match node:
            case Program() | Block() | Region():
                new = self._visit_body(node)
            case For(init) if self.env is not None:
                outer, self.env = self.env, self.env.scope()
                self.env.declare(init)
                new = node.map(self._visit_stmt)
                self.env = outer
            case MultiDecl() if self.env is not None:
                new = node.map(self._visit_decl)
            case If() | For() | While() | DoWhile():
                new = node.map(self._visit_stmt)
            case _:
                new = node.map(self.visit)

        return self.rewrite(new)

    def rewrite(self, node):
        return node

    def _visit_stmt(self, node):
        # Wrap the statements a lone statement comes back as into a block
        new = self.visit(node)

        return Block(new) if isinstance(new, list) else new

    def _visit_decl(self, node):
        # Declare each variable of a statement for those which follow it
        new = self.visit(node)
        self.env.declare(node)

        return new

    def _visit_body(self, node):
        # Open a scope for the declarations in this block
        outer = self.env
        if outer is not None:
            self.env = outer.scope()

        # Visit each statement with the declarations before it
        body, changed = [], False
        for s in node.body:
            new = self.visit(s)

            # Splice in the statements a visit comes back with a list of
            if isinstance(new, list):
                body += new
                changed = True
            else:
                body.append(new)
                changed |= new is not s

            if self.env is not None:
                self.env.declare(s)

        self.env = outer

        return replace(node, body=body) if changed else node


def call(name, args):
    return Call(Var(name), args)


def expr_children(expr):
    # The callee of a call is a name rather than an operand
    if isinstance(expr, Call):
        return tuple(expr.args)
    else:
        return tuple(expr.children())


def walk_expr(expr):
    yield expr

    for c in expr_children(expr):
        yield from walk_expr(c)
