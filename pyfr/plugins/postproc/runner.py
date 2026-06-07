from pyfr.plugins.postproc.adapters import (BoundaryPostProcData,
                                            VolumePostProcData, split_samples)
from pyfr.plugins.postproc.base import get_pp_plugins
from pyfr.util import subclass_where


class PostProcRunner:
    def __init__(self, names, ndims, cfg, export_type):
        self.ndims = ndims
        self.plugins = get_pp_plugins(names, ndims, cfg, export_type)

    def fields(self, public_only=False):
        out = {}
        for pp in self.plugins:
            for fname, varnames in pp.fields.items():
                if public_only and fname.startswith('_'):
                    continue

                out[fname] = varnames

        return out

    def run(self, adapter, public_only=False):
        for pp in self.plugins:
            pp.run(adapter)

        return {n: a for n, a in adapter.fields.items()
                if not (public_only and n.startswith('_'))}

    def run_samples(self, cfg, samples, *, boundary=None, public_only=False):
        if not self.plugins:
            return {}

        from pyfr.solvers.base import BaseSystem
        sname = cfg.get('solver', 'system')
        elementscls = subclass_where(BaseSystem, name=sname).elementscls
        nvars = len(elementscls.privars(self.ndims, cfg))
        pris, grad_pris = split_samples(samples, nvars)
        if boundary is None:
            adapter = VolumePostProcData(cfg, pris, grad_pris)
        else:
            adapter = BoundaryPostProcData(cfg, pris, *boundary, grad_pris)

        return self.run(adapter, public_only=public_only)

    @property
    def needs_grads(self):
        return any(p.needs_grads for p in self.plugins)

    def __bool__(self):
        return bool(self.plugins)
