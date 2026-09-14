import math

from pyfr.dsl.nodes import (Binary, Call, Cast, Float, Index, Int, Number,
                            String, Ternary, Unary, Var, call, expr_children,
                            unwrap_index)


def build_index(base, indices):
    for idx in indices:
        base = Index(base, idx)

    return base


def extract_loop_offset(idx_expr, loop_var):
    match idx_expr:
        # loop_var
        case Var(name) if name == loop_var:
            return 0
        # loop_var + k or k + loop_var
        case Binary('+', Var(name), Int(k)) | Binary('+', Int(k), Var(name)):
            return k if name == loop_var else None
        # loop_var - k
        case Binary('-', Var(name), Int(k)) if name == loop_var:
            return -k
        # A literal index k
        case Int(k):
            return ('static', k)
        # Anything else is not a constant offset from loop_var
        case _:
            return None


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


_dfmin = (lambda x, y: Ternary(Binary('<', x, y), Int(1), Int(0)),
          lambda x, y: Ternary(Binary('<', x, y), Int(0), Int(1)))
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

    def __init__(self, var, dependent_vars=set(), deriv_namer=None, *,
                 array_vars=set(), dependent_arrays=set(), tangent_mode=False,
                 loop_var=None, seed_index=None):
        self.var = var
        self.dependent_vars = dependent_vars
        self.dependent_arrays = dependent_arrays
        self.deriv_namer = deriv_namer or (lambda n: f'd_{n}')
        self.array_vars = array_vars
        self.tangent_mode = tangent_mode
        self.loop_var = loop_var
        self.seed_index = seed_index

    def diff(self, expr):
        D = self.diff
        match expr:
            # Constants
            case Number(_) | String(_):
                return Int(0)
            # Variables
            case Var(name):
                if name == self.var:
                    return Int(1)
                elif name in self.dependent_vars:
                    return Var(self.deriv_namer(name))
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
            # Rewrite pow as **
            case Call(Var('pow'), [base, exp]):
                return D(Binary('**', base, exp))
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
            # Cast
            case Cast(_, inner):
                return D(inner)
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
            # Indices must not depend on the variable
            for idx in indices:
                if self.contains_var(idx):
                    raise NotImplementedError('Cannot differentiate where '
                                              f'index depends on {self.var}')

            # Read the tangent directly from the supplied derivative array
            if self.tangent_mode and aname in self.array_vars:
                darr = Var(self.deriv_namer(aname))
                return build_index(darr, indices)
            # Differentiate input array access with seed index
            elif aname == self.var and self.var in self.array_vars:
                if len(indices) == 1 and self.seed_index is not None:
                    idx, si = indices[0], self.seed_index
                    offset = extract_loop_offset(idx, self.loop_var)
                    soff = extract_loop_offset(si, self.loop_var)

                    # Compare offsets if both can be extracted
                    if offset is not None and soff is not None:
                        return Int(1) if offset == soff else Int(0)
                    # Compare integer indices directly
                    elif isinstance(idx, Int) and isinstance(si, Int):
                        return Int(1) if idx.value == si.value else Int(0)
                    # General case uses runtime comparison
                    else:
                        cond = Binary('==', idx, si)
                        return Ternary(cond, Int(1), Int(0))
                else:
                    return Int(1)
            # Dependent arrays use derivative arrays
            elif aname in self.dependent_vars | self.dependent_arrays:
                darr = Var(self.deriv_namer(aname))
                return build_index(darr, indices)
            else:
                return Int(0)

    def contains_var(self, expr):
        match expr:
            case Var(name):
                return name == self.var or name in self.dependent_vars
            case Index(_, _):
                aname, _ = unwrap_index(expr)
                if aname == self.var and self.var in self.array_vars:
                    return True
                else:
                    return (aname in self.dependent_vars or
                            aname in self.dependent_arrays)
            case _:
                return any(self.contains_var(c) for c in expr_children(expr))
