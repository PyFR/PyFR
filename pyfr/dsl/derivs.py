import math

from pyfr.dsl.nodes import (Binary, Call, Cast, Float, Index, Int, Number,
                            String, Ternary, Unary, Var, call, expr_children,
                            unwrap_index)


def build_index(base, indices):
    for idx in indices:
        base = Index(base, idx)

    return base


def _sq(x):
    return Binary('**', x, Int(2))


def _sqsum(x, y):
    return Binary('+', _sq(x), _sq(y))


def _recip(x):
    return Binary('/', Int(1), x)


def _rsqrt(x):
    return _recip(call('sqrt', [x]))


def _derf(x):
    c = Float(2/math.pi**0.5)

    return Binary('*', c, call('exp', [Unary('-', _sq(x))]))


def _dfabs(x):
    return call('copysign', [Float(1.0), x])


_dfmin = (lambda x, y: Ternary(Binary('<=', x, y), Int(1), Int(0)),
          lambda x, y: Ternary(Binary('<=', x, y), Int(0), Int(1)))
_dfmax = (lambda x, y: Ternary(Binary('<', x, y), Int(0), Int(1)),
          lambda x, y: Ternary(Binary('<', x, y), Int(1), Int(0)))


class Differentiator:
    # Derivatives of single argument functions
    func_derivs = {
        'sin': lambda x: call('cos', [x]),
        'cos': lambda x: Unary('-', call('sin', [x])),
        'tan': lambda x: _recip(_sq(call('cos', [x]))),
        'asin': lambda x: _rsqrt(Binary('-', Int(1), _sq(x))),
        'acos': lambda x: Unary('-', _rsqrt(Binary('-', Int(1), _sq(x)))),
        'atan': lambda x: _recip(Binary('+', Int(1), _sq(x))),
        'sinh': lambda x: call('cosh', [x]),
        'cosh': lambda x: call('sinh', [x]),
        'tanh': lambda x: Binary('-', Int(1), _sq(call('tanh', [x]))),
        'asinh': lambda x: _rsqrt(Binary('+', _sq(x), Int(1))),
        'acosh': lambda x: _rsqrt(Binary('-', _sq(x), Int(1))),
        'atanh': lambda x: _recip(Binary('-', Int(1), _sq(x))),
        'exp': lambda x: call('exp', [x]),
        'exp2': lambda x: Binary('*', call('exp2', [x]), Float(math.log(2))),
        'expm1': lambda x: call('exp', [x]),
        'log': lambda x: _recip(x),
        'log10': lambda x: _recip(Binary('*', x, Float(math.log(10)))),
        'log2': lambda x: _recip(Binary('*', x, Float(math.log(2)))),
        'log1p': lambda x: _recip(Binary('+', Int(1), x)),
        'sqrt': lambda x: _recip(Binary('*', Int(2), call('sqrt', [x]))),
        'cbrt': lambda x: _recip(Binary('*', Int(3), _sq(call('cbrt', [x])))),
        'erf': _derf,
        'erfc': lambda x: Unary('-', _derf(x)),
        'fabs': _dfabs, 'abs': _dfabs,
        'sign': lambda x: Int(0)
    }

    # Derivatives of two argument functions with respect to each argument
    func_derivs2 = {
        'atan2': (lambda y, x: Binary('/', x, _sqsum(x, y)),
                  lambda y, x: Unary('-', Binary('/', y, _sqsum(x, y)))),
        'hypot': (lambda x, y: Binary('/', x, call('hypot', [x, y])),
                  lambda x, y: Binary('/', y, call('hypot', [x, y]))),
        'fmin': _dfmin, 'min': _dfmin,
        'fmax': _dfmax, 'max': _dfmax
    }

    def __init__(self, inputs, dependent_vars, dependent_arrays, tangent):
        self.inputs = inputs
        self.dependent_vars = dependent_vars
        self.dependent_arrays = dependent_arrays

        # Name of the tangent carried by an active name
        self.tangent = tangent

    def _is_active(self, name, arrays=False):
        # Test if the name is an input or depends on one
        if name in self.inputs or name in self.dependent_vars:
            return True
        else:
            return arrays and name in self.dependent_arrays

    def diff(self, expr):
        D = self.diff
        match expr:
            # Constants
            case Number(_) | String(_):
                return Int(0)
            # Variables
            case Var(name):
                if self._is_active(name):
                    return Var(self.tangent(name))
                else:
                    return Int(0)
            # Linear operators
            case Binary('+' | '-' as op, left, right):
                return Binary(op, D(left), D(right))
            # Product rule
            case Binary('*', left, right):
                return Binary('+', Binary('*', D(left), right),
                              Binary('*', left, D(right)))
            # Quotient rule
            case Binary('/', left, right):
                num = Binary('-', Binary('*', D(left), right),
                             Binary('*', left, D(right)))
                return Binary('/', num, Binary('*', right, right))
            # Power rule
            case Binary('**', base, exp):
                if not self.contains_var(exp):
                    p = Binary('**', base, Binary('-', exp, Int(1)))
                    return Binary('*', Binary('*', exp, p), D(base))
                # General power rule
                else:
                    dexp, dbase = D(exp), D(base)
                    t = Binary('+', Binary('*', dexp, call('log', [base])),
                               Binary('*', exp, Binary('/', dbase, base)))
                    return Binary('*', expr, t)
            # Single argument functions via chain rule
            case Call(Var(func), [arg]) if func in self.func_derivs:
                df = self.func_derivs[func](arg)
                return Binary('*', df, D(arg))
            # Two argument functions via chain rule
            case Call(Var(func), [a, b]) if func in self.func_derivs2:
                dfa, dfb = self.func_derivs2[func]
                return Binary('+', Binary('*', dfa(a, b), D(a)),
                              Binary('*', dfb(a, b), D(b)))
            # Unary operators
            case Unary('-', operand):
                return Unary('-', D(operand))
            case Unary('+', operand):
                return D(operand)
            # Ternary
            case Ternary(cond, true, false):
                return Ternary(cond, D(true), D(false))
            # Array indexing
            case Index(_, _) as idx_expr:
                return self._diff_index(idx_expr)
            # Casts to an integer type are piecewise constant
            case Cast(ctype, inner):
                from pyfr.dsl.types import INT_TYPES

                return Int(0) if ctype in INT_TYPES else D(inner)
            # Non-differentiable operators
            case Binary('%', _, _):
                raise NotImplementedError('Modulo not differentiable')
            # Comparison and logical operators are piecewise constant
            case (Binary('<' | '>' | '<=' | '>=' | '==' | '!=', _, _)
                  | Unary('!', _)):
                return Int(0)
            case _:
                raise NotImplementedError('Cannot differentiate '
                                          f'{type(expr).__name__}: {expr}')

    def _diff_index(self, idx_expr):
        aname, indices = unwrap_index(idx_expr)

        # A non-variable base cannot be resolved to a derivative array
        if aname is None:
            if self.contains_var(idx_expr):
                raise NotImplementedError('Cannot differentiate complex index')
            else:
                return Int(0)
        else:
            # Indices must not depend on the inputs
            for idx in indices:
                if self.contains_var(idx):
                    raise NotImplementedError('Cannot differentiate where '
                                              'index depends on the inputs')

            # Read the tangent directly from the derivative array
            if self._is_active(aname, arrays=True):
                return build_index(Var(self.tangent(aname)), indices)
            else:
                return Int(0)

    def contains_var(self, expr):
        match expr:
            case Var(name):
                return self._is_active(name)
            case Index(_, _):
                aname, _ = unwrap_index(expr)
                return self._is_active(aname, arrays=True)
            case _:
                return any(self.contains_var(c) for c in expr_children(expr))
