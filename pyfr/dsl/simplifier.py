from collections import defaultdict

from pyfr.dsl.nodes import (Assign, Binary, Block, Call, Float, For, Int,
                            Number, Region, Ternary, Typedef, Unary, VarDecl,
                            call, expr_children, map_ast)
from pyfr.dsl.types import Environment


def expr_size(expr):
    return 1 + sum(expr_size(c) for c in expr_children(expr))


def expand_power(base, n):
    # Expanding a large base would duplicate it at every factor
    if expr_size(base) > 3:
        return Binary('**', base, Int(n))
    elif n == 2:
        return Binary('*', base, base)
    # Squaring halves the exponent, so the multiply count stays low
    elif n % 2 == 0:
        half = expand_power(base, n // 2)
        return Binary('*', half, half)
    else:
        return Binary('*', base, expand_power(base, n - 1))


def reduce_pow(base, v):
    twov = round(2*v)
    n = abs(int(twov))
    ipart, has_sqrt = divmod(n, 2)

    # Leave exponents which are not half integers or need many multiplies
    if abs(2*v - twov) > 1e-12 or n > 17:
        result = None
    # Anything to the power of zero is one
    elif n == 0:
        result = Int(1)
    # Repeating a large base at every factor costs more than it saves
    elif ipart > 0 and expr_size(base) > 3:
        result = None
    # Expand the integer part and attach the sqrt for the half
    else:
        if ipart == 0:
            result = call('sqrt', [base])
        elif ipart == 1:
            result = base
        else:
            result = expand_power(base, ipart)

        if has_sqrt and ipart > 0:
            result = Binary('*', result, call('sqrt', [base]))

        if twov < 0:
            result = Binary('/', Int(1), result)

    return result


class Simplifier:
    # Region kinds whose statements must not be rearranged
    frozen_regions = frozenset({'fp-precise'})

    def __init__(self, env=None):
        self.env = env or Environment()

    def _kind(self, expr):
        return self.env.type_of(expr).kind

    def _literal(self, c):
        return Float(c) if isinstance(c, float) else Int(c)

    def _const(self, c, kind):
        # Mint a literal of the given kind, or nothing if it is unknown
        if kind == 'float':
            return Float(float(c))
        elif kind == 'int' and c == int(c):
            return Int(int(c))
        else:
            return None

    def _scale(self, c, expr):
        # Attach a coefficient to an expression, dropping a redundant one
        if c < 0:
            return Unary('-', self._scale(-c, expr))
        elif c == 1 and (isinstance(c, int) or self._kind(expr) == 'float'):
            return expr
        else:
            return Binary('*', self._literal(c), expr)

    def _factors(self, expr):
        # Split a product into its numeric coefficient and other factors
        match expr:
            case Number(v):
                return v, []
            case Binary('*', l, r):
                cl, fl = self._factors(l)
                cr, fr = self._factors(r)
                return cl*cr, fl + fr
            case Unary('-', x):
                c, f = self._factors(x)
                return -c, f
            case _:
                return 1, [expr]

    def _product(self, factors):
        expr = None
        for f in factors:
            expr = f if expr is None else Binary('*', expr, f)

        return expr

    def _terms(self, expr, scale=1, terms=None):
        # Collect the terms of a sum, keyed by their non-numeric factors
        terms = {} if terms is None else terms
        match expr:
            case Binary('+', l, r):
                self._terms(l, scale, terms)
                self._terms(r, scale, terms)
            case Binary('-', l, r):
                self._terms(l, scale, terms)
                self._terms(r, -scale, terms)
            case Unary('-', x):
                self._terms(x, -scale, terms)
            case _:
                c, factors = self._factors(expr)
                key = tuple(sorted(map(repr, factors)))
                coeff, t = terms.get(key, (0, self._product(factors)))
                terms[key] = (coeff + scale*c, t)

        return terms

    def _sum(self, terms, ckind):
        expr = None
        for k, (c, t) in terms.items():
            if not k or c == 0:
                continue

            # Fold the sign of each term into the operator joining it
            node = self._scale(abs(c), t)
            if expr is None:
                expr = node if c > 0 else Unary('-', node)
            else:
                expr = Binary('+' if c > 0 else '-', expr, node)

        # Append the numeric term, of the given kind if it stands alone
        c = terms.get((), (0, None))[0]
        if expr is None:
            return self._const(c, ckind)
        elif c == 0:
            return expr
        else:
            return Binary('+' if c > 0 else '-', expr, self._literal(abs(c)))

    def _is_fdiv(self, expr):
        # Test for a floating point division
        is_div = isinstance(expr, Binary) and expr.op == '/'

        return is_div and self._kind(expr) == 'float'

    def _merge_fractions(self, terms):
        # Group the fractional terms by their denominator
        bydenom = defaultdict(list)
        for key, (c, t) in terms.items():
            if c != 0 and self._is_fdiv(t):
                bydenom[repr(t.right)].append(key)

        # a/x + b/x -> (a + b)/x
        for keys in bydenom.values():
            if len(keys) > 1:
                num, denom = {}, terms[keys[0]][1].right
                for k in keys:
                    c, t = terms.pop(k)
                    self._terms(t.left, c, num)

                frac = Binary('/', self._sum(num, 'float'), denom)
                terms[(repr(frac),)] = (1, frac)

    def _simplify_sum(self, expr):
        terms = self._terms(expr)
        self._merge_fractions(terms)

        return self._sum(terms, self._kind(expr)) or expr

    def _cancel(self, c, factors):
        i = 0
        while i < len(factors):
            f = factors[i]
            others = factors[:i] + factors[i + 1:]
            if not self._is_fdiv(f):
                i += 1
                continue

            match f.right:
                # 2*(n/2) -> n
                case Number(v) if v != 0 and c % v == 0:
                    cn, fn = self._factors(f.left)
                    c, factors, i = (c // v)*cn, others + fn, 0
                # d*(n/d) -> n
                case d if d in others:
                    j = others.index(d)
                    cn, fn = self._factors(f.left)
                    c, factors, i = c*cn, others[:j] + others[j + 1:] + fn, 0
                case _:
                    i += 1

        return c, factors

    def _simplify_prod(self, expr):
        c, factors = self._factors(expr)

        # 3*(2*a + 2*b) -> 6*(a + b)
        for i, f in enumerate(factors):
            if isinstance(f, Binary) and f.op in ('+', '-'):
                terms = {k: v for k, v in self._terms(f).items() if v[0] != 0}
                coeffs = {v[0] for v in terms.values()}
                if len(terms) > 1 and len(coeffs) == 1 and coeffs != {1}:
                    k = coeffs.pop()
                    unit = {kk: (1, t) for kk, (_, t) in terms.items()}
                    c, factors[i] = c*k, self._sum(unit, self._kind(f))

        c, factors = self._cancel(c, factors)

        if c == 0:
            return self._const(0, self._kind(expr)) or expr
        elif not factors:
            return self._literal(c)
        else:
            return self._scale(c, self._product(factors))

    def _simplify_div(self, expr):
        kind = self._kind(expr)

        match expr.left, expr.right:
            # Exact integer division folds
            case Int(a), Int(b) if b != 0 and a % b == 0:
                return Int(a // b)
            case Number(a), Number(b) if b != 0 and kind == 'float':
                return Float(a / b)
            case Number(0), _:
                return self._const(0, kind) or expr
            case l, Number(1):
                return l
            # x/x -> 1 for floating point division
            case l, r if l == r and kind == 'float':
                return Float(1.0)
            case _:
                return expr

    def _simplify_pow(self, expr):
        match expr.left, expr.right:
            case _, Number(0):
                return Float(1.0)
            case Int(a), Int(b) if b > 0:
                return Float(float(a**b))
            case base, Number(v):
                return reduce_pow(base, v) or expr
            case _:
                return expr

    def _rewrite(self, expr):
        match expr:
            case Binary('+' | '-', _, _) | Unary('-', _):
                return self._simplify_sum(expr)
            case Binary('*', _, _):
                return self._simplify_prod(expr)
            case Binary('/', _, _):
                return self._simplify_div(expr)
            case Binary('**', _, _):
                return self._simplify_pow(expr)
            case _:
                return expr

    def simplify(self, expr):
        # Simplify the operands before the node itself
        expr = map_ast(expr, self.simplify)
        new = self._rewrite(expr)

        # Refuse any rewrite which would change the type of the result
        if new is not expr and self._kind(new) != self._kind(expr):
            return expr
        else:
            return new

    def simplify_fully(self, expr, max_iter=10):
        # One rewrite can expose another, so iterate to a fixed point
        for _ in range(max_iter):
            if (simplified := self.simplify(expr)) == expr:
                return simplified

            expr = simplified

        return expr

    def simplify_program(self, prog):
        def walk(node):
            match node:
                # Freeze the contents of regions which forbid rearrangement
                case Region(kind) if kind in self.frozen_regions:
                    return node
                # Open a scope for the declarations in this block
                case Block() | Region() | For():
                    outer, self.env = self.env, self.env.scope()
                    node = map_ast(node, walk)
                    self.env = outer
                    return node
                case VarDecl(_, vtype, name):
                    node = map_ast(node, walk)
                    self.env.define(name, vtype)
                    return node
                case Typedef(old, new):
                    self.env.define(new, old)
                    return node
                case Binary() | Unary() | Call() | Ternary() | Assign():
                    return self.simplify_fully(node)
                case _:
                    return map_ast(node, walk)

        return walk(prog)
