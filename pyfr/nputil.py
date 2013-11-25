# -*- coding: utf-8 -*-

import ast
import ctypes as ct
import itertools as it
import re

import numpy as np


def npaligned(shape, dtype, alignb=32):
    nbytes = np.prod(shape)*np.dtype(dtype).itemsize
    buf = np.zeros(nbytes + alignb, dtype=np.uint8)
    off = -buf.ctypes.data % alignb

    return buf[off:nbytes + off].view(dtype).reshape(shape)


_npeval_syms = {
    '__builtins__': None,
    'exp': np.exp, 'log': np.log,
    'sin': np.sin, 'asin': np.arcsin,
    'cos': np.cos, 'acos': np.arccos,
    'tan': np.tan, 'atan': np.arctan, 'atan2': np.arctan2,
    'abs': np.abs, 'pow': np.power, 'sqrt': np.sqrt,
    'pi': np.pi}


def npeval(expr, locals):
    # Ensure the expression does not contain invalid characters
    if not re.match(r'[A-Za-z0-9 \t\n\r.,+\-*/^%()]+$', expr):
        raise ValueError('Invalid characters in expression')

    # Disallow access to object attributes
    objs = '|'.join(it.chain(_npeval_syms, locals))
    if re.search(r'(%s|\))\s*\.' % objs, expr):
        raise ValueError('Invalid expression')

    # Allow '^' to be used for exponentiation
    expr = expr.replace('^', '**')

    return eval(expr, _npeval_syms, locals)


def fuzzysort(arr, idx, dim=0, tol=1e-6):
    # Extract our dimension and argsort
    arrd = arr[dim]
    srtdidx = sorted(idx, key=arrd.__getitem__)

    i, ix = 0, srtdidx[0]
    for j, jx in enumerate(srtdidx[1:], start=1):
        if arrd[jx] - arrd[ix] >= tol:
            if j - i > 1:
                srtdidx[i:j] = fuzzysort(arr, srtdidx[i:j], dim + 1, tol)
            i, ix = j, jx

    if i != j:
        srtdidx[i:] = fuzzysort(arr, srtdidx[i:], dim + 1, tol)

    return srtdidx


def range_eval(expr):
    r = []

    for prt in map(lambda s: s.strip(), expr.split('+')):
        try:
            # Parse parts of the form [x, y, z, ...]
            if prt.startswith('['):
                r.extend(float(l) for l in ast.literal_eval(prt))
            # Parse parts of the form range(start, stop, n)
            else:
                m = re.match(r'range\((.*?),(.*?),(.*?)\)$', prt)
                s, e, n = m.groups()

                r.extend(np.linspace(float(s), float(e), int(n)))
        except (AttributeError, SyntaxError, ValueError):
            raise ValueError('Invalid range')

    return r


_ctype_map = {
    np.int32: 'int', np.uint32: 'unsigned int',
    np.int64: 'long long', np.uint64: 'unsigned long long',
    np.float32: 'float', np.float64: 'double'}


def npdtype_to_ctype(dtype):
    return _ctype_map[np.dtype(dtype).type]


_ctypestype_map = {
    np.int32: ct.c_int32, np.uint32: ct.c_uint32,
    np.int64: ct.c_int64, np.uint64: ct.c_uint64,
    np.float32: ct.c_float, np.float64: ct.c_double}


def npdtype_to_ctypestype(dtype):
    # Special-case None which otherwise expands to np.float
    if dtype is None:
        return None

    return _ctypestype_map[np.dtype(dtype).type]
