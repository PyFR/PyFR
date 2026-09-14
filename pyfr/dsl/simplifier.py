from pyfr.dsl.nodes import (Assign, Binary, Call, Float, Int, Number, Region,
                            Ternary, Unary, Var, call, expr_children, map_ast)


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

    def _is_const(self, expr):
        match expr:
            case Number(v):
                return v
            case _:
                return None

    def _get_const(self, expr):
        match expr:
            case Number(v):
                return (v, None)
            case Binary('*', c, rest) if (cv := self._is_const(c)) is not None:
                return (cv, rest)
            case _:
                return None

    def _strip_coeff(self, expr):
        match expr:
            case Binary('*', c, rest) if (cv := self._is_const(c)) is not None:
                return (cv, rest)
            case Binary('*', left, right):
                n, stripped = self._strip_coeff(left)
                if n != 1:
                    return (n, Binary('*', stripped, right))
                else:
                    return (1, expr)
            case _:
                return (1, expr)

    def _make_const(self, c):
        return Float(c) if isinstance(c, float) else Int(c)

    def _scale_expr(self, c, expr):
        if c == 1:
            return expr
        elif c == 0:
            return Int(0)
        elif expr:
            return Binary('*', self._make_const(c), expr)
        else:
            return self._make_const(c)

    def _simplify_add(self, left, right):
        match left, right:
            # x + x -> 2*x
            case l, r if l == r:
                return Binary('*', Int(2), l)
            # 3*x + x -> 4*x
            case Binary('*', c, x), r if (cv := self._is_const(c)) and x == r:
                return Binary('*', self._make_const(cv + 1), r)
            # x + 3*x -> 4*x
            case l, Binary('*', c, x) if (cv := self._is_const(c)) and l == x:
                return Binary('*', self._make_const(cv + 1), l)
            # x*y + y*x -> 2*(x*y)
            case Binary('*', a, b), Binary('*', c, d) if a == d and b == c:
                return Binary('*', Int(2), Binary('*', a, b))
            # a/x + b/x -> (a + b)/x
            case Binary('/', a, d1), Binary('/', b, d2) if d1 == d2:
                return Binary('/', Binary('+', a, b), d1)
            # 2*x + 3*x -> 5*x
            case _:
                nl, sl = self._strip_coeff(left)
                nr, sr = self._strip_coeff(right)
                if sl == sr and (nl != 1 or nr != 1):
                    return self._scale_expr(nl + nr, sl)
                else:
                    return Binary('+', left, right)

    def _simplify_sub(self, left, right):
        match left, right:
            # x*y - y*x -> 0
            case Binary('*', a, b), Binary('*', c, d) if a == d and b == c:
                return Int(0)
            # a/x - b/x -> (a - b)/x
            case Binary('/', a, d1), Binary('/', b, d2) if d1 == d2:
                return Binary('/', Binary('-', a, b), d1)
            case _:
                return Binary('-', left, right)

    def _simplify_mul(self, left, right):
        # Merge a constant factor with the coefficient on the other side
        match left, right:
            # (2*x)*3 -> 6*x
            case Binary('*', c1, x), _:
                v1, v2 = self._is_const(c1), self._is_const(right)
                if v1 is not None and v2 is not None:
                    return self._scale_expr(v1*v2, x)
            # 2*(3*x) -> 6*x
            case _, Binary('*', c2, x):
                v1, v2 = self._is_const(left), self._is_const(c2)
                if v1 is not None and v2 is not None:
                    return self._scale_expr(v1*v2, x)

        # Gather coefficients when the left is itself a scaled expression
        match right:
            # (2*a)*(3*y) -> 6*(a*y)
            case Binary('*', k, y) if (kv := self._is_const(k)) is not None:
                if c1 := self._get_const(left):
                    new_c = c1[0]*kv
                    if c1[1] is None:
                        return self._scale_expr(new_c, y)
                    else:
                        return Binary('*', self._scale_expr(new_c, c1[1]), y)
            # 3*(2*a + 2*b) -> 6*(a + b)
            case Binary('+', ladd, radd):
                cl, cr = self._get_const(ladd), self._get_const(radd)
                if cl and cr and cl[0] == cr[0]:
                    k, new_sum = cl[0], Binary('+', cl[1], cr[1])
                    if c1 := self._get_const(left):
                        new_c = c1[0]*k
                        if c1[1] is None:
                            return self._scale_expr(new_c, new_sum)
                        else:
                            sc = self._scale_expr(new_c, c1[1])
                            return Binary('*', sc, new_sum)

        return Binary('*', left, right)

    def simplify(self, expr):
        match expr:
            case Binary(op, left, right):
                left, right = self.simplify(left), self.simplify(right)
                match op, left, right:
                    # Identity and zero rules for addition/subtraction
                    case '+' | '-', l, Number(0):
                        return l
                    case '+', Number(0), r:
                        return r
                    case '-', Number(0), r:
                        return Unary('-', r)
                    case '-', l, r if l == r:
                        return Int(0)
                    # Sign simplifications
                    case '-', l, Unary('-', r):
                        return Binary('+', l, r)
                    case '+', l, Unary('-', r):
                        return Binary('-', l, r)
                    case '+', Unary('-', l), r:
                        return Binary('-', r, l)
                    # Zero and identity rules for multiplication/division
                    case '*' | '/', Number(0), _:
                        return Int(0)
                    case '*', _, Number(0):
                        return Int(0)
                    case '*', Number(1), r:
                        return r
                    case '*', l, Number(1):
                        return l
                    case '*', Number(-1), r:
                        return Unary('-', r)
                    case '*', l, Number(-1):
                        return Unary('-', l)
                    case '*', l, Unary('-', r):
                        return Unary('-', Binary('*', l, r))
                    case '*', Unary('-', l), r:
                        return Unary('-', Binary('*', l, r))
                    # Constant folding for integers
                    case '+' | '-' | '*', Int(a), Int(b):
                        return Int({'+': a + b, '-': a - b, '*': a*b}[op])
                    case '/', Int(a), Int(b) if b != 0 and a % b == 0:
                        return Int(a // b)
                    # Power simplifications
                    case '**', Int(a), Int(b) if b >= 0:
                        return Int(a ** b)
                    case '**', base, Number(v) as e:
                        return reduce_pow(base, v) or Binary('**', base, e)
                    # Constant folding for mixed int/float
                    case '+' | '-' | '*', Number(a), Number(b):
                        return Float({'+': a + b, '-': a - b, '*': a*b}[op])
                    # Inexact integer division truncates in C; leave it be
                    case '/', Int(), Int():
                        return Binary(op, left, right)
                    case '/', Number(a), Number(b) if b != 0:
                        return Float(a/b)
                    # Cancellation rules
                    case '/', l, r if l == r:
                        return Int(1)
                    case '*', l, Binary('/', n, d) if l == d:
                        return n
                    case '*', Binary('/', n, d), r if d == r:
                        return n
                    # Advanced simplifications
                    case '*', _, _:
                        return self._simplify_mul(left, right)
                    case '+', _, _:
                        return self._simplify_add(left, right)
                    case '-', _, _:
                        return self._simplify_sub(left, right)
                    case _:
                        return Binary(op, left, right)
            case Unary('-', Number(0)):
                return Int(0)
            case Unary('-', Unary('-', x)):
                return self.simplify(x)
            case Unary('-', Number(v)):
                return self._make_const(-v)
            case Unary('-', Binary('-', a, b)):
                return Binary('-', self.simplify(b), self.simplify(a))
            case Unary(op, operand):
                return Unary(op, self.simplify(operand))
            case Call(Var(func), args):
                return call(func, [self.simplify(a) for a in args])
            case Ternary(c, t, f):
                sc = self.simplify(c)
                st, sf = self.simplify(t), self.simplify(f)
                return Ternary(sc, st, sf)
            case Assign(left, right):
                return Assign(left, self.simplify(right))
            case _:
                return expr

    def simplify_fully(self, expr, max_iter=10):
        # One rewrite can expose another, so iterate to a fixed point
        for _ in range(max_iter):
            if (simplified := self.simplify(expr)) == expr:
                return simplified

            expr = simplified

        return expr

    def simplify_program(self, prog):
        def walk(node):
            # Freeze the contents of regions which forbid rearrangement
            if isinstance(node, Region) and node.kind in self.frozen_regions:
                return node
            elif isinstance(node, (Binary, Unary, Call, Ternary, Assign)):
                return self.simplify_fully(node)
            else:
                return map_ast(node, walk)

        return walk(prog)
