import numpy as np

from pyfr.plugins.fields.base import get_field_providers
from pyfr.snapshot import FieldInfo


class SampleView:
    # A per-key (etype/itype/'points') lens onto a SnapshotSample, handed to a
    # field provider.  Provides simple attribute access to pris/grad_pris/
    # normals/wall_dist/state — backed by sample (mutable per step) and region
    # (static geometry).  Lives here, with the runner that constructs it,
    # rather than in snapshot.py (avoids a snapshot↔runner import cycle).
    def __init__(self, sample, etype):
        self.sample = sample
        self.region = sample.region
        self.snap = sample.snap
        self.cfg = sample.region.config
        self.ndims = sample.region.ndims
        self.etype = etype
        # Per-view write target; FieldRunner collects into sample.fields.
        self.fields = {}

    @property
    def ploc(self):
        return self.sample.ploc[self.etype]

    @property
    def pris(self):
        return self.sample.pris[self.etype]

    @property
    def grad_pris(self):
        return self.sample.grad_pris[self.etype]

    @property
    def normals(self):
        return getattr(self.region, 'normals', {}).get(self.etype)

    @property
    def min_upt_wall_dist_approx(self):
        return getattr(self.region, 'wall_dist', {}).get(self.etype)

    @property
    def state(self):
        return self.snap.state

    @property
    def nvars(self):
        return len(self.pris)

    @property
    def has_grads(self):
        return self.grad_pris is not None


class FieldRunner:
    # Bundle of dep-resolved field providers + aggregated metadata.  Caches
    # the topo-sorted provider list and exposes the union of their .fields
    # dicts, so callers (writers, renderer, sampler) can register the
    # outputs they're about to publish before any run() is invoked.
    def __init__(self, names, ndims, cfg, export_type):
        self.ndims = ndims
        self.plugins = get_field_providers(names, ndims, cfg, export_type)

    def fields(self, public_only=False):
        out = {}
        for p in self.plugins:
            for fname, varnames in p.fields.items():
                if public_only and fname.startswith('_'):
                    continue

                out[fname] = varnames

        return out

    def run_on_sample(self, sample, public_only=False):
        # One pair of loops: per-key (etype/itype/'points') × per-provider.
        # Providers run in topo order (transformers first via kind, then
        # producers in dep order) so a transformer's in-place mutation of
        # sample.ploc / sample.pris is visible to downstream producers.
        for et in sample.region.etypes:
            view = SampleView(sample, et)
            for p in self.plugins:
                p.run(view)
            # Collect this view's outputs into the sample's per-(et, fname)
            # dict so subsequent consumers can index by both.
            for fname, arr in view.fields.items():
                if public_only and fname.startswith('_'):
                    continue
                sample.fields[et, fname] = arr

        # Register provider outputs in sample.fields_meta so consumers iterate
        # one unified registry (alongside snap's primitives + grads + aux).
        for p in self.plugins:
            for fname, varnames in p.fields.items():
                if public_only and fname.startswith('_'):
                    continue
                if fname in sample.fields_meta:
                    continue
                sample.fields_meta[fname] = FieldInfo(
                    name=fname, kind='point', ncomps=len(varnames),
                    dtype=np.dtype(sample.snap.dtype), source='provider',
                    components=tuple(varnames))

        return sample.fields

    @property
    def needs_grads(self):
        return any(p.needs_grads for p in self.plugins)

    def __bool__(self):
        return bool(self.plugins)
