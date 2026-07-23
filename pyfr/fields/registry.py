from importlib.resources import files
import re

import numpy as np

from pyfr.fluids import get_fluid
from pyfr.fluids.base import _host_syms
from pyfr.inifile import Inifile

GRAD_RE = re.compile(r'\bgrad_(\w+?)_([xyz])\b')
GEOM_RE = re.compile(r'\b(n_[xyz]|wall_dist)\b')


class FieldRegistry:
    def __init__(self, cfg, ndims):
        self.ndims = ndims
        self.fluid = get_fluid(cfg, ndims)

        # Constants; bare-name ones double as expression symbols
        self._allc = cfg.items_as('constants', float)
        self.c = {k: v for k, v in self._allc.items() if k.isidentifier()}

        # Load the shipped field tables
        tbl = Inifile((files('pyfr.fields') / 'tables'
                       / 'fields.txt').read_text())

        self._entries = {}
        for sect in tbl.sections():
            deps = [d.strip() for d in tbl.get(sect, 'deps', '').split(',')
                    if d.strip()]

            consts = {}
            for kv in tbl.get(sect, 'consts', '').split(','):
                if kv.strip():
                    sym, key = (s.strip() for s in kv.split(':'))
                    consts[sym] = key

            expr = (tbl.get(sect, f'expr-{ndims}d', '')
                    or tbl.get(sect, 'expr', ''))
            if not expr:
                continue

            comps = [e.strip() for e in expr.split('|')]

            grad = (tbl.get(sect, f'grad-{ndims}d', '')
                    or tbl.get(sect, 'grad', '')).strip() or None
            if grad and len(comps) != 1:
                raise ValueError(f'Field {sect!r}: gradients may only be '
                                 'declared for scalar fields')

            self._entries[sect] = (deps, consts, comps, grad)

        self._flatten_entries()

        # Ad-hoc expression cache for non-table field requests
        self._adhoc = {}

    def _flatten_entries(self):
        # Inline references between entries, merging deps and consts
        ident = [n for n, v in self._entries.items()
                 if n.isidentifier() and len(v[2]) == 1]
        if not ident:
            return

        pat = re.compile(r'\b({})\b'.format('|'.join(map(re.escape, ident))))

        for _ in range(5):
            changed = False

            for name, (deps, consts, comps, grad) in list(
                    self._entries.items()):
                def sub(m):
                    nonlocal changed

                    if m[1] == name:
                        raise ValueError(f'Field {name!r} references itself')

                    changed = True

                    rdeps, rconsts, rcomps = self._entries[m[1]][:3]
                    deps.extend(d for d in rdeps if d not in deps)
                    for sym, key in rconsts.items():
                        if consts.setdefault(sym, key) != key:
                            raise ValueError(
                                f'Constant symbol {sym!r} conflict in '
                                f'field {name!r}'
                            )

                    return f'({rcomps[0]})'

                ncomps = [pat.sub(sub, c) for c in comps]
                self._entries[name] = (deps, consts, ncomps, grad)

            if not changed:
                break

    # Inline the differential of a fluid quantity in privar-gradient form
    def _quantity_grad(self, name, dim, stack=()):
        fluid = self.fluid

        if name in stack:
            raise ValueError(f'Cyclic differential for quantity {name!r}')

        de = fluid.diff_expr(name)
        if isinstance(de, list):
            raise ValueError(f"'grad_{name}_{dim}': quantity {name!r} is "
                             'not a scalar')

        # Map differential symbols of privar quantities to their gradients
        dmap = {}
        for pn, (qn, comp) in zip(fluid.privars, fluid.pri_map):
            dmap[f'd_{qn}' if comp is None else f'd_{qn}[{comp}]'] = pn

        def sub(m):
            key = m[0]
            if key in dmap:
                return f'grad_{dmap[key]}_{dim}'
            if m[2] is None and fluid.diffable(m[1]):
                return f'({self._quantity_grad(m[1], dim, (*stack, name))})'

            raise ValueError(
                f"'grad_{name}_{dim}': differential of {name!r} requires "
                f'{key!r} which cannot be expressed in primitive-variable '
                'gradients'
            )

        de = re.sub(r'\bd_(\w+)(\[(?:\d+)\])?', sub, de)

        if re.search(r'\bdu\b', de):
            raise ValueError(
                f"'grad_{name}_{dim}': differential of {name!r} references "
                'the conservative state directly'
            )

        return de

    # Rewrite non-primitive gradient symbols down to privar gradients
    def _expand_grads(self, expr):
        pv = set(self.fluid.privars)

        for _ in range(5):
            if all(m[1] in pv for m in GRAD_RE.finditer(expr)):
                return expr

            def sub(m):
                base, dim = m[1], m[2]

                if base in pv:
                    return m[0]

                if base in self._entries and not base.startswith('_'):
                    grad = self._entries[base][3]
                    if grad:
                        return '(' + grad.replace('{d}', dim) + ')'

                    raise ValueError(
                        f"'{m[0]}': field {base!r} does not declare a "
                        "gradient (no 'grad' entry in its definition)"
                    )

                if self.fluid.diffable(base):
                    return f'({self._quantity_grad(base, dim)})'

                raise ValueError(
                    f"'{m[0]}': gradients are available for primitive "
                    f"variables ({', '.join(self.fluid.privars)}), fields "
                    "with a declared 'grad' line, and fluid quantities "
                    'with a declared differential'
                )

            expr = GRAD_RE.sub(sub, expr)

        raise ValueError('Unable to resolve gradients in field expression')

    # Inline scalar table entries into an expression
    def _inline_entries(self, expr):
        ent = [n for n, v in self._entries.items()
               if n.isidentifier() and len(v[2]) == 1
               and not n.startswith('_')]
        if not ent:
            return expr

        p = '|'.join(re.escape(n) for n in ent)
        for _ in range(5):
            expr, n = re.subn(rf'\b({p})\b',
                              lambda m: f'({self._inline(m[1])})', expr)
            if not n:
                break

        return expr

    # Synthesise an entry for an ad-hoc field expression
    def _adhoc_entry(self, expr):
        if expr in self._adhoc:
            return self._adhoc[expr]

        e = self._inline_entries(expr)
        e = self._expand_grads(e)
        e = self._inline_entries(e)

        # Scalar fluid quantities referenced by the expression
        pv = set(self.fluid.privars)
        deps = [n for n in self.fluid.quantities()
                if n not in pv and n.isidentifier()
                and self.fluid.quantity_shape(n) == 1
                and re.search(rf'\b{re.escape(n)}\b', e)]

        self._adhoc[expr] = ent = (deps, {}, [e], None)

        return ent

    def _resolve(self, names):
        # Fluid quantities required by the requested names
        fnames = []

        for n in names:
            if n in self._entries and not n.startswith('_'):
                fnames.extend(self._entries[n][0])
            elif self.fluid.provides(n):
                if self.fluid.quantity_shape(n) != 1:
                    raise ValueError(f'Field {n!r} is not a scalar')

                fnames.append(n)
            else:
                fnames.extend(self._adhoc_entry(n)[0])

        return list(dict.fromkeys(fnames))

    def fields(self, names):
        self._resolve(names)

        return {n: len(self._entries[n][2]) if n in self._entries else 1
                for n in names}

    def _inline(self, n):
        deps, consts, comps, _ = self._entries[n]
        e = comps[0]

        if consts:
            p = '|'.join(re.escape(s) for s in consts)

            def sub(m):
                try:
                    return repr(self._allc[consts[m[1]]])
                except KeyError:
                    raise ValueError(f'Field {n!r} requires constant '
                                     f'{consts[m[1]]!r}') from None

            e = re.sub(rf'\b({p})\b', sub, e)

        return e

    def expand(self, expr):
        # Inline scalar table entries and resolve declared gradients
        expr = self._inline_entries(expr)
        expr = self._expand_grads(expr)
        expr = self._inline_entries(expr)

        # Rewrite scalar fluid quantities; privars keep their meaning
        pv = set(self.fluid.privars)
        fq = [n for n in self.fluid.quantities()
              if n not in pv and n.isidentifier()
              and self.fluid.quantity_shape(n) == 1]

        fnames = set()
        if fq:
            p = '|'.join(re.escape(n) for n in fq)

            def sub(m):
                fnames.add(m[1])
                return f'{m[1]}_qf'

            expr = re.sub(rf'\b({p})\b', sub, expr)

        return expr, sorted(fnames)

    def _comps_of(self, n):
        if n in self._entries:
            return self._entries[n][2]
        elif self.fluid.provides(n):
            return []
        else:
            return self._adhoc_entry(n)[2]

    def needs_grads(self, names):
        return any(GRAD_RE.search(c) for n in names for c in self._comps_of(n))

    def needs_geom(self, names):
        return any(GEOM_RE.search(c) for n in names for c in self._comps_of(n))

    def eval_quantities(self, names, pris):
        fluid = self.fluid

        seed = fluid.pri_seed(list(pris))
        cons = fluid.pri_to_con(list(pris))
        vals = fluid.eval(', '.join(names), cons, seed=seed) if names else {}

        # Broadcast constant quantities over the sample points
        shape = np.shape(pris[0])
        return {n: np.asarray(v) if np.ndim(v) else np.full(shape, v)
                for n, v in vals.items()}

    def evaluate(self, names, samples, geom=None):
        fnames = self._resolve(names)

        if geom is None and self.needs_geom(names):
            raise ValueError('Requested fields need a boundary context')

        # Primitive variables and, if present, their gradients
        pv = self.fluid.privars
        nv = len(pv)
        ns = dict(zip(pv, samples[:nv]))

        if geom:
            ns |= geom

        if len(samples) > nv:
            for i, f in enumerate(pv):
                for d, x in enumerate('xyz'[:self.ndims]):
                    ns[f'grad_{f}_{x}'] = samples[nv + i*self.ndims + d]

        # Evaluate any required fluid quantities, seeded with the state
        vals = self.eval_quantities(fnames, [samples[i] for i in range(nv)])

        out = {}
        for n in names:
            if n in self._entries or not self.fluid.provides(n):
                deps, consts, comps, _ = (self._entries.get(n)
                                          or self._adhoc_entry(n))

                ens = _host_syms | self.c | ns
                ens |= {d: vals[d] for d in deps}
                for sym, key in consts.items():
                    try:
                        ens[sym] = self._allc[key]
                    except KeyError:
                        raise ValueError(
                            f'Field {n!r} requires constant {key!r}'
                        ) from None

                try:
                    res = [eval(c, dict(ens)) for c in comps]
                except NameError as e:
                    raise ValueError(
                        f'Unknown symbol in field {n!r}: {e.name}'
                    ) from None
                v = np.stack(res, axis=-1) if len(res) > 1 else res[0]
            else:
                v = vals[n]

            # Broadcast constant quantities over the sample points
            out[n] = (np.asarray(v) if np.ndim(v)
                      else np.full(samples[0].shape, v))

        return out
