from collections import defaultdict
import fnmatch
from importlib.resources import files
import re

from pyfr.exprs import subst_expr_vars
from pyfr.util import expand_braces


KEYWORDS = ('field', 'derived', 'derived-hidden', 'set')


def parse_stats(spec):
    pat = r'([\w*-]+)(?:\{([\w*, -]+)\}([\w*-]*))?'

    # Parse each package, expanding any brace enumerations
    pkgs, npats = [], 0
    for m in re.finditer(pat, spec):
        npats += 1
        pkgs.extend(expand_braces(m[0]))

    # The specification must be exactly these patterns, comma separated
    rem = re.sub(r'\s', '', re.sub(pat, '', spec))
    if not pkgs or rem != ','*(npats - 1):
        raise ValueError(f'Invalid statistics specification: {spec}')

    return pkgs


def _parse_stanzas(text):
    stanza = None

    for ln, line in enumerate(text.splitlines(), start=1):
        # Normalise whitespace and skip comments
        norm = ' '.join(line.split())
        if not norm or norm.startswith('#'):
            continue

        # Indented lines continue the current stanza body
        if line[0] in ' \t':
            if stanza is None:
                raise ValueError(f'Unexpected continuation on line {ln}')
            stanza[-1] += f' {norm}'
        else:
            if stanza:
                yield stanza

            m = re.fullmatch(r'([a-z-]+) ([^\s=]+)(?: requires '
                             r'([^=]+?))?(?: ?= ?(.+))?', norm)
            if m is None or m[1] not in KEYWORDS:
                raise ValueError(f'Invalid stanza header on line {ln}')

            stanza = list(m.groups(''))

    if stanza:
        yield stanza


def _parse_table(text):
    registry, sets = {}, {}
    requires = defaultdict(list)

    for kw, name, reqs, body in _parse_stanzas(text):
        body = body.strip()

        if reqs:
            requires[name].extend(reqs.split())

        if kw == 'set':
            sets[name] = body.split()
        elif name in registry:
            raise ValueError(f'Duplicate statistics quantity: {name}')
        else:
            registry[name] = (kw, body)

    return registry, sets, requires


def name_tokens(expr):
    # Derivative symbols reference their underlying quantity
    pat = r'\blap_(\w+)|\bgrad_(\w+)_[xyz]\b|([A-Za-z][\w-]*(?<!-))'

    for m in re.finditer(pat, expr):
        yield m[m.lastindex]


class StatsTables:
    def __init__(self, tables, ndims, kind):
        self.ndims = ndims
        self.registry, self.sets = {}, {}
        self.requires = defaultdict(list)

        # Each table is a resource directory inside a solver package
        for t in tables:
            path = files(t).joinpath('stats', f'{kind}-{ndims}d.txt')
            reg, sets, req = _parse_table(path.read_text())

            # Later tables may add to, but never redefine, quantities
            if dups := self.registry.keys() & reg.keys():
                raise ValueError('Duplicate statistics quantities: '
                                 f'{', '.join(sorted(dups))}')

            self.registry |= reg
            self.sets |= sets
            for n, r in req.items():
                self.requires[n].extend(r)

    def dependencies(self, names):
        # Requested names plus everything they transitively reference
        deps, pending = dict.fromkeys(names), list(names)

        while pending:
            group, expr = self.registry[pending.pop()]

            # Field integrands reference the instantaneous solution
            if group == 'field':
                continue

            for tok in name_tokens(expr):
                if tok in self.registry and tok not in deps:
                    deps[tok] = None
                    pending.append(tok)

        return deps

    def check_requires(self, names, consts):
        for n in names:
            for r in self.requires.get(n, []):
                if r not in consts:
                    raise ValueError(f'Statistics quantity {n} requires '
                                     f'the {r} constant')


    def validate(self):
        dpat = r'\b(?:lap_(\w+)|grad_(\w+)_[xyz])\b'

        # Only fields and hidden quantities may be differentiated
        diffable = {n for n, (g, e) in self.registry.items()
                    if g in ('field', 'derived-hidden')}

        for name, (group, expr) in self.registry.items():
            # Field integrands may reference solver gradients directly
            if group == 'field':
                continue

            for m in re.finditer(dpat, expr):
                if group != 'derived' or (m[1] or m[2]) not in diffable:
                    raise ValueError(f'{group} {name} may not take the '
                                     f'derivative {m[0]}')

    def resolve(self, spec):
        names, snames = {}, []

        def add_set(s):
            if s not in self.sets:
                raise ValueError(f'Unknown statistics set: {s}')

            # Sets may nest; visited tracking renders this cycle-safe
            if s in snames:
                return

            snames.append(s)
            for t in self.sets[s]:
                # A self-named member refers to the quantity
                if t in self.sets and t != s:
                    add_set(t)
                else:
                    names[t] = None

        for name in parse_stats(spec):
            # Wildcard patterns select over the available sets
            if '*' in name:
                matches = fnmatch.filter(self.sets, name)
                if not matches:
                    raise ValueError(f'Unmatched statistics pattern: {name}')

                for s in matches:
                    add_set(s)
            elif name in self.sets:
                add_set(name)
            elif name in self.registry:
                names[name] = None
            else:
                raise ValueError(f'Unknown statistics package: {name}')

        return names, snames

    def classify(self, seen):
        # Derived quantities evaluable from the fields alone
        dpat = (r'\b(?:lap_|grad_\w+_[xyz]\b|grid_|'
                r'norm_[xyz]\b|boundary_dist\b)')
        alg = set()

        # Table order is topological for derived-to-derived references
        for name, (group, expr) in self.registry.items():
            if name not in seen or group == 'field':
                continue

            # Differential if it takes derivatives or needs the geometry
            if re.search(dpat, expr):
                continue

            # Or if it references a non-algebraic derived quantity
            for tok in name_tokens(expr):
                if (tok in self.registry and tok != name and
                    self.registry[tok][0] != 'field' and tok not in alg):
                    break
            else:
                alg.add(name)

        return alg

    def expand(self, cfg, spec):
        self.validate()
        names, snames = self.resolve(spec)
        seen = self.dependencies(names)

        # Validate constant requirements over the sets and the closure
        c = cfg.items_as('constants', float)
        self.check_requires((*snames, *seen), c)

        # Bin the closure by group in table order
        groups = {g: {} for g in KEYWORDS[:-1]}
        for name, (group, expr) in self.registry.items():
            if name in seen:
                groups[group][name] = expr

        return groups, self.classify(seen)

    def exprs(self, elementscls, cfg, names):
        if missing := set(names) - self.registry.keys():
            raise ValueError('Unknown derived quantities: '
                             f'{', '.join(sorted(missing))}')

        seen = self.dependencies(names)

        # Validate constant requirements over the closure
        c = cfg.items_as('constants', float)
        self.check_requires(seen, c)

        # Lower the expressions; constants may contain dashes
        aux = elementscls.auxvars(self.ndims, cfg)
        cpat = '|'.join(map(re.escape, c))
        csub = lambda e: re.sub(rf'(?<![\w-])({cpat})(?![\w-])',
                                lambda m: str(c[m[1]]), e)
        lower = lambda e: f'({csub(subst_expr_vars(e, aux))})'

        # Requested quantities are outputs; the remainder support them
        wanted = {n: lower(e) for n, (g, e) in self.registry.items()
                  if n in names and n in seen}
        support = {n: lower(e) for n, (g, e) in self.registry.items()
                   if n in seen and n not in names}

        return wanted, support


def soln_exprs(elementscls, ndims, cfg, names):
    tabs = StatsTables(elementscls.stats_tables(cfg), ndims, 'soln')
    return tabs.exprs(elementscls, cfg, names)
