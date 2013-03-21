# -*- coding: utf-8 -*-

import numpy as np


def npaligned(shape, dtype, alignb=32):
    nbytes = np.prod(shape)*np.dtype(dtype).itemsize
    buf = np.zeros(nbytes + alignb, dtype=np.uint8)
    off = -buf.ctypes.data % alignb

    return buf[off:nbytes + off].view(dtype).reshape(shape)


_npeval_syms = {'__builtins__': None,
                'exp': np.exp, 'log': np.log,
                'sin': np.sin, 'asin': np.arcsin,
                'cos': np.cos, 'acos': np.arccos,
                'tan': np.tan, 'atan': np.arctan, 'atan2': np.arctan2,
                'abs': np.abs, 'pow': np.power, 'sqrt': np.sqrt,
                'pi': np.pi}


def npeval(expr, locals):
    # Allow '^' to be used for exponentiation
    expr = expr.replace('^', '**')

    return eval(expr, _npeval_syms, locals)


_range_eval_syms = {'__builtins__': None,
                    'range': lambda s,e,n: list(np.linspace(s, e, n))}


def range_eval(expr):
    return [float(t) for t in eval(expr, _range_eval_syms, None)]

_ctype_map = {np.float32: 'float', np.float64: 'double'}


def npdtype_to_ctype(dtype):
    return _ctype_map[np.dtype(dtype).type]
