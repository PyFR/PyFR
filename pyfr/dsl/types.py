from dataclasses import dataclass
from functools import reduce

from pyfr.dsl.derivs import Differentiator
from pyfr.dsl.nodes import (Assign, Binary, Call, Cast, DslCall, DslVar,
                            Float, Index, Int, MultiDecl, Postfix, Ternary,
                            Unary, Var, VarDecl, unwrap_index)


@dataclass(frozen=True, slots=True)
class Type:
    name: str
    kind: str | None
    rank: int = 0


# Integer types by conversion rank, with the DSL's own index type widest
INT_TYPES = {t.name: t for t in [
    Type('int8_t', 'int', 1), Type('uint8_t', 'int', 1),
    Type('int16_t', 'int', 2), Type('uint16_t', 'int', 2),
    Type('int', 'int', 3), Type('int32_t', 'int', 3),
    Type('uint32_t', 'int', 3), Type('int64_t', 'int', 4),
    Type('uint64_t', 'int', 4), Type('ssize_t', 'int', 4),
    Type('size_t', 'int', 4), Type('ixdtype_t', 'int', 5)
]}

# Floating point types by conversion rank, with the DSL's own type widest
FLOAT_TYPES = {t.name: t for t in [
    Type('float', 'float', 1), Type('double', 'float', 2),
    Type('fpdtype_t', 'float', 3)
]}

SCALAR_TYPES = INT_TYPES | FLOAT_TYPES

INT = INT_TYPES['int']
DOUBLE = FLOAT_TYPES['double']
UNKNOWN = Type('unknown', None)

# Math functions which always return a floating point value
FLOAT_FNS = (frozenset(Differentiator.func_derivs) |
             frozenset(Differentiator.func_derivs2) | {'pow', 'copysign'})

# Operators whose result is an integer whatever their operands
INT_OPS = frozenset('% << >> & | ^ == != < <= > >= && ||'.split())


def common_type(a, b):
    # Apply the usual arithmetic conversions to a pair of operand types
    if a.kind == b.kind:
        return a if a.rank >= b.rank else b
    elif a.kind == 'float':
        return a
    elif b.kind == 'float':
        return b
    else:
        return UNKNOWN


class Environment:
    def __init__(self, default=UNKNOWN):
        self.enclosing = None
        self.default = default
        self.names = {}
        self.decls = {}

    def scope(self):
        env = Environment(self.default)
        env.enclosing = self

        return env

    def _scope_of(self, name):
        # Search the innermost scope first, then those enclosing it
        env = self
        while env is not None and name not in env.names:
            env = env.enclosing

        return env

    def lookup(self, name):
        env = self._scope_of(name)
        return env.names[name] if env is not None else None

    def lookup_decl(self, name):
        env = self._scope_of(name)
        return env.decls[name] if env is not None else None

    def resolve(self, ctype):
        return self.lookup(ctype) or SCALAR_TYPES.get(ctype, UNKNOWN)

    def define(self, name, ctype, sizes=None):
        self.names[name] = self.resolve(ctype)
        self.decls[name] = (ctype, sizes)

    def declare(self, stmt):
        # Define the names introduced by a declaration statement
        match stmt:
            case VarDecl(_, vtype, name, sizes, _):
                self.define(name, vtype, sizes)
            case MultiDecl(decls):
                for d in decls:
                    self.define(d.name, d.vtype, d.sizes)

    def get(self, name):
        return self.lookup(name) or self.default

    def type_of(self, expr):
        match expr:
            case Int() | DslVar() | DslCall():
                return INT
            case Float():
                return DOUBLE
            case Var(name):
                return self.get(name)
            case Index():
                return self.get(unwrap_index(expr)[0])
            case Cast(ctype, _):
                return self.resolve(ctype)
            case Binary(op, _, _) if op in INT_OPS:
                return INT
            case Binary('**', l, r):
                t = common_type(self.type_of(l), self.type_of(r))
                return common_type(DOUBLE, t)
            case Binary(',', _, r):
                return self.type_of(r)
            case Binary(_, l, r) | Ternary(_, l, r):
                return common_type(self.type_of(l), self.type_of(r))
            case Unary('!', _):
                return INT
            case Unary(_, x) | Postfix(_, x) | Assign(x, _):
                return self.type_of(x)
            case Call(Var(name), args) if name in FLOAT_FNS:
                types = (self.type_of(a) for a in args)
                return reduce(common_type, types, DOUBLE)
            case _:
                return self.default
