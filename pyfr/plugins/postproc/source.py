from pyfr.plugins.postproc.adapters import (BoundaryPostProcData,
                                            PostProcData)
from pyfr.plugins.postproc.derived import TableDerivedPostProc
from pyfr.util import subclass_where


def get_source(prefix, cfg, stats, ndims):
    return subclass_where(BaseDataSource, prefix=prefix)(cfg, stats, ndims)


class BaseDataSource:
    prefix = None
    adapters = None
    plugins = {}

    def __init__(self, cfg, stats, ndims):
        self.cfg = cfg
        self.stats = stats
        self.ndims = ndims

    def adapter(self, soln, kind, samples, ploc, *args):
        return self.adapters[kind].from_soln(soln, samples, ploc, *args)

    def pipeline(self, names, kind, cfg=None, want=None, akind=None,
                 provided=None):
        return PostProcPipeline(self, names, kind, cfg, want, akind, provided)

    def quantities(self, names, export_type, cfg=None, want=None,
                   provided=None):
        cfg = cfg or self.cfg

        def resolve(name):
            if (cls := self.plugins.get(name)) is not None:
                return cls(self, cfg, export_type, want)
            else:
                return TableDerivedPostProc(name, self, cfg, export_type, want)

        # Resolve the explicitly requested plugins and derived tables
        qs = []
        for name in dict.fromkeys(names):
            if (q := resolve(name)).fields:
                qs.append(q)

        # Requested fields nothing else supplies become table derivations
        have = set(provided or ()) | {f for q in qs for f in q.fields}
        for name in sorted((want or set()) - have):
            if (q := resolve(name)).fields:
                qs.append(q)

        return qs

    def prepare(self, mesh, soln, pp_plugins):
        pass


class SolnDataSource(BaseDataSource):
    prefix = 'soln'
    adapters = {'volume': PostProcData, 'boundary': BoundaryPostProcData}


class PostProcPipeline:
    def __init__(self, source, names, kind, cfg=None, want=None, akind=None,
                 provided=None):
        self.source = source
        self.akind = akind or kind
        self.plugins = plugins = source.quantities(names, kind, cfg, want,
                                                   provided)

        self.needs_grads = any(pp.needs_grads for pp in plugins)
        self.fields = {n: v for pp in plugins for n, v in pp.fields.items()}

    def __call__(self, soln, samples, ploc, *args):
        data = self.source.adapter(soln, self.akind, samples, ploc, *args)
        for pp in self.plugins:
            pp.run(data)

        return data.fields
