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
            self._entries[sect] = (deps, consts, comps)

        self._flatten_entries()

    def _flatten_entries(self):
        # Inline references between entries, merging deps and consts
        ident = [n for n, v in self._entries.items()
                 if n.isidentifier() and len(v[2]) == 1]
        if not ident:
            return

        pat = re.compile(r'\b({})\b'.format('|'.join(map(re.escape, ident))))

        for _ in range(5):
            changed = False

            for name, (deps, consts, comps) in list(self._entries.items()):
                def sub(m):
                    nonlocal changed

                    if m[1] == name:
                        raise ValueError(f'Field {name!r} references itself')

                    changed = True

                    rdeps, rconsts, rcomps = self._entries[m[1]]
                    deps.extend(d for d in rdeps if d not in deps)
                    for sym, key in rconsts.items():
                        if consts.setdefault(sym, key) != key:
                            raise ValueError(
                                f'Constant symbol {sym!r} conflict in '
                                f'field {name!r}'
                            )

                    return f'({rcomps[0]})'

                ncomps = [pat.sub(sub, c) for c in comps]
                self._entries[name] = (deps, consts, ncomps)

            if not changed:
                break

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
                raise ValueError(f'Unknown field {n!r}')

        return list(dict.fromkeys(fnames))

    def fields(self, names):
        self._resolve(names)

        return {n: len(self._entries[n][2]) if n in self._entries else 1
                for n in names}

    def _inline(self, n):
        deps, consts, comps = self._entries[n]
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
        # Inline scalar table entries into the expression
        ent = [n for n, v in self._entries.items()
               if n.isidentifier() and len(v[2]) == 1
               and not n.startswith('_')]
        if ent:
            p = '|'.join(re.escape(n) for n in ent)
            for _ in range(5):
                expr, n = re.subn(rf'\b({p})\b',
                                  lambda m: f'({self._inline(m[1])})', expr)
                if not n:
                    break

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

    def needs_grads(self, names):
        return any(GRAD_RE.search(c)
                   for n in names if n in self._entries
                   for c in self._entries[n][2])

    def needs_geom(self, names):
        return any(GEOM_RE.search(c)
                   for n in names if n in self._entries
                   for c in self._entries[n][2])

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
            if n in self._entries:
                deps, consts, comps = self._entries[n]

                ens = _host_syms | self.c | ns
                ens |= {d: vals[d] for d in deps}
                for sym, key in consts.items():
                    try:
                        ens[sym] = self._allc[key]
                    except KeyError:
                        raise ValueError(
                            f'Field {n!r} requires constant {key!r}'
                        ) from None

                res = [eval(c, dict(ens)) for c in comps]
                v = np.stack(res, axis=-1) if len(res) > 1 else res[0]
            else:
                v = vals[n]

            # Broadcast constant quantities over the sample points
            out[n] = (np.asarray(v) if np.ndim(v)
                      else np.full(samples[0].shape, v))

        return out
