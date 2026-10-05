from functools import cached_property
import re

import numpy as np

from pyfr.exprs import npeval
from pyfr.fields import expand_stats_fields
from pyfr.plugins.postproc.adapters import BoundaryPostProcData, PostProcData
from pyfr.plugins.postproc.base import BasePostProcPlugin
from pyfr.plugins.postproc.source import BaseDataSource
from pyfr.stats import eval_exports, tavg_exprs
from pyfr.stats.provider import name_tokens


class TavgDataMixin:
    def __init__(self, source, pvars, ploc, *args):
        self.texprs = source.texprs

        # Keep the stored rows under their row names
        self._stored = pvars

        super().__init__(source, self._pvars(), ploc, *args)

    @cached_property
    def porder(self):
        return self.cfg.getint('solver', 'order')

    @property
    def aliases(self):
        return self.texprs.aliases

    def _pvars(self):
        # Expose each field under its name without the average prefix
        pvars = {n.removeprefix('avg-'): v for n, v in self._stored.items()}

        # Deduplicated moments resolve through their alias
        pvars |= {c: pvars[s] for c, s in self.aliases.items()}

        # Primitive (or grad_X_Y) name -> index in soln.fields for tavg
        for name, expr in self.texprs.avgs.items():
            if (var := expr.strip('() \t')).isidentifier():
                pvars.setdefault(var, pvars[name])

        return pvars

    def avg(self, name):
        # Resolve canonical moment names through the alias map
        name = self.aliases.get(name, name)

        return self._stored[f'avg-{name}']

    def grad_avg(self, name):
        # Gradient components are stacked contiguously per field
        name = self.aliases.get(name, name)
        dims = 'xyz'[:self.ndims]

        return [self._stored[f'grad_{name}_{x}'] for x in dims]

    def lap_avg(self, name):
        name = self.aliases.get(name, name)

        return self._stored[f'lap_{name}']

    def grid_h(self, i):
        return self._stored[('grid_hmin', 'grid_hmax')[i]]

    def __getitem__(self, name):
        # Gradients of the mean fields stand in for those of the primitives
        if m := re.fullmatch(r'grad_(\w+?)_([xyz])', name):
            return self.grad_avg(m[1])['xyz'.index(m[2])]
        else:
            # Mean fields stand in for the primitives at export time
            try:
                return self.avg(name)
            except KeyError:
                raise RuntimeError('Postproc on averages requires mean '
                                   'statistics for all primitive variables')


class TavgPostProcData(TavgDataMixin, PostProcData):
    pass


class TavgBoundaryPostProcData(TavgDataMixin, BoundaryPostProcData):
    pass


class StatsFinalisePostProc(BasePostProcPlugin):
    name = 'stats'
    systems = '.*'
    dimensions = '2|3'
    export_types = '.*'

    def __init__(self, source, cfg, export_type=None, kind=None, want=None):
        super().__init__(source, cfg, export_type, kind, want)

        # Derive the output expressions from the configuration
        te = source.texprs
        self.hidden = te.hidden
        self.derived = dict(te.derived)

        # Quantities needing surface geometry, directly or transitively
        bpat = r'\b(norm_[xyz]|boundary_dist)\b'
        onb = {}
        for n, e in {**self.hidden, **self.derived}.items():
            deps = any(map(onb.get, name_tokens(e)))
            onb[n] = bool(re.search(bpat, e)) or deps
        self.boundary_only = {n for n, b in onb.items() if b}

        # Boundary-only quantities are unavailable off a boundary export
        if kind != 'boundary':
            self.derived = {n: e for n, e in self.derived.items()
                            if n not in self.boundary_only}

        # Restrict to the requested outputs plus their dependencies
        if want is not None:
            keep = self._dep_closure(want)
            self.derived = {n: e for n, e in self.derived.items()
                            if n in keep}
            self.fields = {n: [n] for n in self.derived if n in want}
        else:
            self.fields = {n: [n] for n in self.derived}

    def _dep_closure(self, want):
        # Close the requested names over their expression dependencies
        keep = {n for n in self.derived if n in want}
        pending = list(keep)
        while pending:
            for tok in name_tokens(self.derived.get(pending.pop(), '')):
                if tok in self.derived and tok not in keep:
                    keep.add(tok)
                    pending.append(tok)

        return keep

    @property
    def needs_grads(self):
        return any(re.search(r'\b(grad|lap)_', e)
                   for e in self.derived.values())

    @property
    def needs_gridh(self):
        return any(re.search(r'\bgrid_h', e) for e in self.derived.values())

    def derived_fields(self, fields):
        subs = {n.removeprefix('avg-'): v for n, v in fields.items()
                if n.startswith('avg-')}

        with np.errstate(divide='ignore', invalid='ignore'):
            return {n: npeval(e, subs) for n, e in self.hidden.items()}

    def _process(self, data):
        out = eval_exports(self.derived, data, self.hidden)
        data.fields |= out


class TavgDataSource(BaseDataSource):
    prefix = 'tavg'
    adapters = {'volume': TavgPostProcData,
                'boundary': TavgBoundaryPostProcData}
    plugins = {'stats': StatsFinalisePostProc}

    @cached_property
    def texprs(self):
        cfgsect = self.stats.get('tavg', 'cfg-section')

        # Re-derive the expression lists from the stored configuration
        return tavg_exprs(self.cfg, cfgsect, self.ndims, self.elementscls)

    def pvar_names(self, rownames, groups):
        return list(rownames)

    def to_pvars(self, block, groups):
        return block.astype(float)

    def prepare(self, mesh, soln, pp_plugins):
        # Averaged data lacks stored gradients; compute them on demand
        return expand_stats_fields(mesh, soln, pp_plugins, self.elementscls)
