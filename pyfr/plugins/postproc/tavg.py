from functools import cached_property
import re

import numpy as np

from pyfr.exprs import npeval
from pyfr.fields import expand_stats_fields
from pyfr.plugins.common import get_elementscls
from pyfr.plugins.postproc.adapters import BoundaryPostProcData, PostProcData
from pyfr.plugins.postproc.base import BasePostProcPlugin
from pyfr.plugins.postproc.source import BaseDataSource
from pyfr.stats import eval_algebraic_avgs, eval_exports, tavg_exprs
from pyfr.stats.provider import name_tokens


class TavgDataMixin:
    def __init__(self, soln, samples, ploc):
        self.soln = soln
        self._samples = samples
        self.ploc = ploc
        self.fields = {}

    @classmethod
    def from_soln(cls, soln, samples, ploc, *args):
        return cls(soln, samples, ploc, *args)

    @property
    def cfg(self):
        return self.soln.config

    @property
    def nvars(self):
        return len(self.soln.fields)

    @property
    def has_grads(self):
        # The layout expansion appends rows for the gradients
        return any(n.startswith('grad_') for n in self.soln.layout)

    @cached_property
    def elementscls(self):
        return get_elementscls(self.cfg)

    @cached_property
    def porder(self):
        return self.cfg.getint('solver', 'order')

    @cached_property
    def texprs(self):
        soln = self.soln
        cfgsect = soln.stats.get('tavg', 'cfg-section')

        # Re-derive the expression lists from the stored configuration
        return tavg_exprs(soln.config, cfgsect, self.ndims, self.elementscls)

    @property
    def aliases(self):
        return self.texprs.aliases

    @cached_property
    def _rows(self):
        # Row indices come from the named layout of the data
        return {n: i for i, n in enumerate(self.soln.layout)}

    def funavg(self, name):
        # Evaluate an algebraic derived quantity from the fields
        te = self.texprs
        fields = {n: self.avg(n) for n in te.avgs}

        return eval_algebraic_avgs(te, fields)[name]

    def avg(self, name):
        # Resolve canonical moment names through the alias map
        name = self.aliases.get(name, name)

        return self._samples[self._rows[f'avg-{name}']]

    def grad_avg(self, name):
        # Gradient components are stacked contiguously per field
        name = self.aliases.get(name, name)
        i = self._rows[f'grad_{name}_x']

        return self._samples[i:i + self.ndims]

    def lap_avg(self, name):
        name = self.aliases.get(name, name)

        return self._samples[self._rows[f'lap_{name}']]

    def grid_h(self, i):
        return self._samples[self._rows['grid_hmin'] + i]

    @cached_property
    def pris(self):
        privars = self.elementscls.privars(self.ndims, self.cfg)

        # Mean fields stand in for the primitives at export time
        try:
            return [self.avg(v) for v in privars]
        except ValueError:
            raise RuntimeError('Postproc on averages requires mean '
                               'statistics for all primitive variables')

    @cached_property
    def grad_pris(self):
        privars = self.elementscls.privars(self.ndims, self.cfg)

        # Gradients of the mean fields stand in for those of the primitives
        return [self.grad_avg(v) for v in privars]


class TavgPostProcData(TavgDataMixin, PostProcData):
    pass


class TavgBoundaryPostProcData(TavgDataMixin, BoundaryPostProcData):
    def __init__(self, soln, samples, ploc, elementscls, spts, finfo):
        super().__init__(soln, samples, ploc)

        self._elementscls = elementscls
        self._spts = spts
        self._finfo = finfo


class StatsFinalisePostProc(BasePostProcPlugin):
    name = 'stats'
    systems = '.*'
    dimensions = '2|3'
    export_types = '.*'

    def __init__(self, source, cfg, export_type=None, want=None):
        super().__init__(source, cfg, export_type, want)

        cfgsect = source.stats.get('tavg', 'cfg-section')

        # Derive the output expressions from the configuration
        te = tavg_exprs(cfg, cfgsect, source.ndims, get_elementscls(cfg))
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
        if export_type != 'boundary':
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
        # Boundary-only quantities require the surface geometry
        if hasattr(data, 'normals'):
            derived = self.derived
        else:
            derived = {n: e for n, e in self.derived.items()
                       if n not in self.boundary_only}

        out = eval_exports(derived, data, self.hidden)
        data.fields |= out


class TavgDataSource(BaseDataSource):
    prefix = 'tavg'
    adapters = {'volume': TavgPostProcData,
                'boundary': TavgBoundaryPostProcData}
    plugins = {'stats': StatsFinalisePostProc}

    def prepare(self, mesh, soln, pp_plugins):
        # Averaged data lacks stored gradients; compute them on demand
        expand_stats_fields(mesh, soln, pp_plugins,
                            get_elementscls(soln.config))
