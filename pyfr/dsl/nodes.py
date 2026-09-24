from dataclasses import dataclass, field, fields, is_dataclass


@dataclass(slots=True)
class Stmt:
    comments: tuple = field(default=(), kw_only=True)


@dataclass(slots=True)
class Program:
    body: list


@dataclass(slots=True)
class VarDecl(Stmt):
    const: bool
    vtype: str
    name: str
    sizes: list | None = None
    init: object = None


@dataclass(slots=True)
class Typedef(Stmt):
    old: str
    new: str


@dataclass(slots=True)
class ArrayInit:
    values: list


@dataclass(slots=True)
class MultiDecl(Stmt):
    decls: list


@dataclass(slots=True)
class Block(Stmt):
    body: list


@dataclass(slots=True)
class Region(Stmt):
    kind: str
    body: list
    args: tuple = ()


@dataclass(slots=True)
class If(Stmt):
    cond: object
    then: object
    else_: object = None


@dataclass(slots=True)
class While(Stmt):
    cond: object
    body: object


@dataclass(slots=True)
class For(Stmt):
    init: object
    cond: object
    step: object
    body: object


@dataclass(slots=True)
class DoWhile(Stmt):
    body: object
    cond: object


@dataclass(slots=True)
class Break(Stmt):
    pass


@dataclass(slots=True)
class Continue(Stmt):
    pass


@dataclass(slots=True)
class ExprStmt(Stmt):
    expr: object


@dataclass(slots=True)
class Comment(Stmt):
    kind: str
    text: str


@dataclass(slots=True)
class Number:
    value: int | float
    suffix: str = ''
    ishex: bool = False


@dataclass(slots=True)
class Int(Number):
    pass


@dataclass(slots=True)
class Float(Number):
    pass


@dataclass(slots=True)
class String:
    value: str


@dataclass(slots=True)
class Var:
    name: str


@dataclass(slots=True)
class Binary:
    op: str
    left: object
    right: object


@dataclass(slots=True)
class Unary:
    op: str
    operand: object


@dataclass(slots=True)
class Postfix:
    op: str
    operand: object


@dataclass(slots=True)
class Ternary:
    cond: object
    true: object
    false: object


@dataclass(slots=True)
class Assign:
    left: object
    right: object


@dataclass(slots=True)
class Call:
    func: object
    args: list


@dataclass(slots=True)
class Index:
    array: object
    index: object


@dataclass(slots=True)
class Cast:
    ctype: str
    expr: object


@dataclass(slots=True)
class DslCall:
    name: str
    args: list


@dataclass(slots=True)
class DslVar:
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


def _ast_children(node):
    for f in fields(node):
        v = getattr(node, f.name)
        for c in (v if isinstance(v, list) else [v]):
            if is_dataclass(c):
                yield c


def walk_ast(node):
    yield node

    for c in _ast_children(node):
        yield from walk_ast(c)


def walk_ast_regions(node, kinds=frozenset()):
    yield node, kinds

    # Nodes beneath a region carry its kind in their context
    if isinstance(node, Region):
        kinds = kinds | {node.kind}

    for c in _ast_children(node):
        yield from walk_ast_regions(c, kinds)


def region_kinds(node):
    return frozenset(n.kind for n in walk_ast(node) if isinstance(n, Region))


# Replace regions of the given kinds with plain blocks
def unwrap_regions(node, kinds):
    def unwrap(n):
        n = map_ast(n, unwrap)
        if isinstance(n, Region) and n.kind in kinds:
            return Block(n.body, comments=n.comments)
        else:
            return n

    return unwrap(node)


def map_ast(node, t):
    new = _map_ast(node, t)

    # Carry any leading comment trivia across the rebuild
    if new is not node and isinstance(new, Stmt):
        new.comments = node.comments

    return new


def _map_ast(node, t):
    # Leaves and nodes with no children are handed back untouched
    if not any(_ast_children(node)):
        return node

    # Rebuild the node with each child field passed through t
    kw = {}
    for f in fields(node):
        v = getattr(node, f.name)
        if isinstance(v, list):
            kw[f.name] = [t(c) if is_dataclass(c) else c for c in v]
        elif is_dataclass(v):
            kw[f.name] = t(v)
        else:
            kw[f.name] = v

    return type(node)(**kw)


def call(name, args):
    return Call(Var(name), args)


def expr_children(expr):
    # The callee of a call is a name rather than an operand
    if isinstance(expr, Call):
        return tuple(expr.args)
    else:
        return tuple(_ast_children(expr))
