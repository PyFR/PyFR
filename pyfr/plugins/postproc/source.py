from functools import cached_property

from pyfr.fields import con_block_to_pri
from pyfr.plugins.common import get_elementscls
from pyfr.plugins.postproc.adapters import (BoundaryPostProcData,
                                            PostProcData)
from pyfr.plugins.postproc.base import BasePostProcPlugin
from pyfr.plugins.postproc.derived import TableDerivedPostProc
from pyfr.readers.native import group_names
from pyfr.util import subclass_where, subclasses


def get_source(prefix, cfg, stats, ndims):
    return subclass_where(BaseDataSource, prefix=prefix)(cfg, stats, ndims)


class BaseDataSource:
    prefix = None
    adapters = None

    def __init__(self, cfg, stats, ndims):
        self.cfg = cfg
        self.stats = stats
        self.ndims = ndims

    @cached_property
    def elementscls(self):
        return get_elementscls(self.cfg)

    @property
    def plugins(self):
        # Named plugins self-declare their data source via source_prefix
        return {cls.name: cls for cls in subclasses(BasePostProcPlugin)
                if cls.source_prefix == self.prefix}

    def adapter(self, kind, names, samples, ploc, *args):
        pvars = {n: samples[:, i] for i, n in enumerate(names)}
        return self.adapters[kind](self, pvars, ploc, *args)

    def pipeline(self, names, export_type, cfg=None, want=None, kind=None,
                 provided=None):
        return PostProcPipeline(self, names, export_type, cfg, want, kind,
                                provided)

    def quantities(self, names, export_type, kind, cfg=None, want=None,
                   provided=None):
        cfg = cfg or self.cfg

        def resolve(name):
            if (cls := self.plugins.get(name)) is not None:
                return cls(self, cfg, export_type, kind, want)
            else:
                return TableDerivedPostProc(name, self, cfg, export_type,
                                            kind, want)

        # Resolve the explicitly requested plugins and derived tables.
        # Transforms mutate the data in place and declare no fields.
        qs = []
        for name in dict.fromkeys(names):
            if (q := resolve(name)).transform or q.fields:
                qs.append(q)

        # Requested fields nothing else supplies become table derivations
        have = set(provided or ()) | {f for q in qs for f in q.fields}
        for name in sorted((want or set()) - have):
            if (q := resolve(name)).fields:
                qs.append(q)

        # Transforms run before any field derivation that consumes their
        # output (stable sort preserves request order within each group)
        qs.sort(key=lambda q: not q.transform)

        return qs

    def prepare(self, mesh, soln, pp_plugins):
        return []

    def pvar_names(self, rownames, groups):
        pass

    def to_pvars(self, block, groups):
        pass


class SolnDataSource(BaseDataSource):
    prefix = 'soln'
    adapters = {'volume': PostProcData, 'boundary': BoundaryPostProcData}

    def pvar_names(self, rownames, groups):
        privars = self.elementscls.privars(self.ndims, self.cfg)

        return privars + group_names(privars, self.ndims, groups)

    def to_pvars(self, block, groups):
        block = block.astype(float).swapaxes(0, 1)

        # Convert to primitive variables at the solution points
        pvars = con_block_to_pri(self.elementscls, self.cfg, self.ndims,
                                 block, groups)

        return pvars.swapaxes(0, 1)


class PostProcPipeline:
    def __init__(self, source, names, export_type, cfg=None, want=None,
                 kind=None, provided=None):
        self.source = source
        self.kind = kind or export_type
        self.plugins = plugins = source.quantities(names, export_type,
                                                   self.kind, cfg, want,
                                                   provided)

        self.needs_grads = any(pp.needs_grads for pp in plugins)
        self.fields = {n: v for pp in plugins for n, v in pp.fields.items()}

    def __call__(self, names, samples, ploc, *args):
        data = self.source.adapter(self.kind, names, samples, ploc, *args)
        for pp in self.plugins:
            pp.run(data)

        return data
