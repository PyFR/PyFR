import re
import warnings

import numpy as np

from pyfr.fluids.transport import BaseTransport
from pyfr.util import subclass_where


class Quantity:
    def __init__(self, deps, expr=None, device=None, host=None, diff=None):
        self.deps = deps
        self.expr = expr
        self.device = device
        self.host = host
        self.diff = diff


_host_syms = {
    'sqrt': np.sqrt, 'fabs': np.abs, 'pow': np.power,
    'min': np.minimum, 'max': np.maximum, 'fmax': np.maximum,
    'log': np.log, 'exp': np.exp
}


class BaseFluid:
    name = None

    def __init__(self, cfg, ndims):
        self.ndims = ndims
        self.c = cfg.items_as('constants', float)

        prec = cfg.get('backend', 'precision', 'double')
        fpdtype = np.float32 if prec == 'single' else np.float64
        self.fpdtype_max = float(np.finfo(fpdtype).max)

        self._read_cfg(cfg)

        # Resolve the transport model
        tname = cfg.get('solver', 'transport', '')
        if not tname:
            vc = cfg.get('solver', 'viscosity-correction', '')
            if vc:
                warnings.warn('[solver] viscosity-correction is deprecated; '
                              'use transport = constant | sutherland')

            tname = 'sutherland' if vc == 'sutherland' else 'constant'

        self.transport = subclass_where(BaseTransport, name=tname)(cfg, self)

        self._quantities = {}
        self._register_quantities()

    def _read_cfg(self, cfg):
        pass

    def _register_quantities(self):
        self._register_state()
        self.transport.register(self)
        self._validate()

    def _validate(self):
        for name, q in self._quantities.items():
            for d in q.deps:
                if d not in self._quantities:
                    raise ValueError(
                        f'Quantity {name!r} depends on {d!r} which fluid '
                        f'{self.name!r} with transport '
                        f'{self.transport.name!r} does not provide'
                    )

    # Fluids ride along in tplargs which are pickled for kernel caching;
    # the quantity emitters are closures and so are rebuilt on unpickle
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != '_quantities'}

    def __setstate__(self, state):
        self.__dict__.update(state)

        self._quantities = {}
        self._register_quantities()

    def _register_quantity(self, name, deps, expr=None, device=None,
                           host=None, diff=None):
        self._quantities[name] = Quantity(deps, expr, device, host, diff)

    def _closure(self, names):
        order, seen = [], set()

        def visit(n):
            if n not in seen:
                seen.add(n)

                try:
                    q = self._quantities[n]
                except KeyError:
                    raise ValueError(f'Fluid {self.name!r} does not provide '
                                     f'quantity {n!r}') from None

                for d in q.deps:
                    visit(d)

                order.append(n)

        for n in names.split(','):
            visit(n.strip())

        return order

    def _suffix(self, expr, suffix):
        if not suffix:
            return expr

        qnames = '|'.join(self._quantities)

        return re.sub(rf'\b({qnames})\b', rf'\g<1>{suffix}', expr)

    def decl(self, u, names, suffix=''):
        lines = []

        for n in self._closure(names):
            q = self._quantities[n]

            if q.device:
                lines.append(q.device(u, suffix))
            elif isinstance(expr := q.expr(u), list):
                lines.append(f'fpdtype_t {n}{suffix}[{len(expr)}];')
                lines.extend(f'{n}{suffix}[{i}] = {self._suffix(e, suffix)};'
                             for i, e in enumerate(expr))
            else:
                lines.append(f'fpdtype_t {n}{suffix} = '
                             f'{self._suffix(expr, suffix)};')

        return '\n'.join(lines)

    # Seed values for graph evaluation from a known primitive state
    def pri_seed(self, pris):
        return {}

    # Primitive variables as (quantity, component) pairs
    @property
    def pri_map(self):
        raise NotImplementedError

    def provides(self, name):
        return name in self._quantities

    def diffable(self, name):
        q = self._quantities.get(name)

        return q is not None and q.diff is not None

    def diff_expr(self, name, u='u', du='du'):
        q = self._quantities.get(name)

        return q.diff(u, du) if q is not None and q.diff else None

    def _diff_closure(self, names):
        order, seen = [], set()

        def visit(n):
            if n in seen:
                return
            seen.add(n)

            q = self._quantities.get(n)
            if q is None or q.diff is None:
                raise ValueError(
                    f'Fluid {self.name!r} does not declare a differential '
                    f'for quantity {n!r}'
                )

            expr = q.diff('u', 'du')
            for e in expr if isinstance(expr, list) else [expr]:
                for m in re.finditer(r'\bd_(\w+)\b', e):
                    visit(m[1])

            order.append(n)

        for n in names.split(','):
            visit(n.strip())

        return order

    # Differentials of quantities given a seed differential of the state
    def eval_diff(self, names, u, du, seed=None):
        order = self._diff_closure(names)

        # Values referenced by the differential expressions
        qnames = '|'.join(self._quantities)
        vnames = list(order)
        for n in order:
            expr = self._quantities[n].diff('u', 'du')
            for e in expr if isinstance(expr, list) else [expr]:
                for m in re.finditer(rf'\b({qnames})\b', e):
                    if m[1] not in vnames:
                        vnames.append(m[1])

        vns = dict(self.eval(', '.join(vnames), u, seed=seed))

        env = {'__builtins__': None, 'u': u, 'du': du} | _host_syms | vns
        for n in order:
            q = self._quantities[n]

            expr = q.diff('u', 'du')
            if isinstance(expr, list):
                env[f'd_{n}'] = [eval(e, dict(env)) for e in expr]
            else:
                env[f'd_{n}'] = eval(expr, dict(env))

        return {n.strip(): env[f'd_{n.strip()}'] for n in names.split(',')}

    # Differentials of the primitive variables (the con-to-pri Jacobian
    # applied to du; du may carry a leading axis, e.g. one per dimension)
    def diff_pri(self, cons, dcons):
        names = []
        for qn, comp in self.pri_map:
            if qn not in names:
                names.append(qn)

        d = self.eval_diff(', '.join(names), cons, dcons)

        return [d[qn] if comp is None else d[qn][comp]
                for qn, comp in self.pri_map]

    def quantities(self):
        return list(self._quantities)

    def quantity_shape(self, name):
        q = self._quantities[name]

        if q.expr and isinstance(e := q.expr('u'), list):
            return len(e)
        else:
            return 1

    def eval(self, names, u, seed=None):
        ns = dict(seed) if seed else {}

        for n in self._closure(names):
            if n in ns:
                continue

            q = self._quantities[n]

            if q.host:
                ns[n] = q.host(u, ns)
            elif isinstance(expr := q.expr('u'), list):
                ns[n] = [self._heval(e, u, ns) for e in expr]
            else:
                ns[n] = self._heval(expr, u, ns)

        return {n.strip(): ns[n.strip()] for n in names.split(',')}

    def _heval(self, expr, u, ns):
        return eval(expr, {'__builtins__': None, 'u': u} | _host_syms | ns)
