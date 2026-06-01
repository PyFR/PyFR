import numpy as np


# ---------------------------------------------------------------------------
# SnapshotSample: solution evaluated on a region's geometry for one snap.
# Per-call object; created by region.sample(snap), used for one render step
# (or one offline export pass), then dropped.  Holds:
#   - ploc: starts as an ALIAS to region.ploc; transformer providers (NIRF)
#     call make_ploc_writable() to copy-on-write before mutating, so the
#     region's static body-frame geometry is never compounded across calls.
#   - pris, grad_pris: per-call (populated by region._compute_sample(...))
#   - fields: aux passthrough + provider outputs keyed by (etype, fname)
#   - fields_meta: dict[name, FieldInfo] — full registry of emittable fields.
#     Inherits snap.fields (primitives + grads + aux) at construction; provider
#     entries get added by FieldRunner.run_on_sample as they run.
# ---------------------------------------------------------------------------

class SnapshotSample:
    def __init__(self, region, snap):
        self.region = region
        self.snap = snap

        # Aux passthrough + provider outputs, keyed by (etype/itype, fname)
        self.fields = {}

        # Field metadata registry — inherit snap's (primitives + grads + aux);
        # FieldRunner extends with provider entries as they run.
        self.fields_meta = dict(snap.fields)

        # ploc starts aliased to the region's static coords; a transformer
        # provider calls make_ploc_writable() before mutating.
        self.ploc = region.ploc

        # Region populates pris + grad_pris + aux pass-through into self.
        region._compute_sample(self, snap)

    def field_array(self, etype, info):
        # Per-key array lookup for a FieldInfo, dispatching on info.source.
        # Primitives stack their component arrays from sample.pris (indexed by
        # snap.pris_names — privars for soln-form, stored_fields for stored-
        # form).  Gradients stack from sample.grad_pris.  Aux + provider are
        # direct sample.fields lookups (None if absent — e.g. PointsSnapshot-
        # Region which doesn't pass aux through).
        names = self.snap.pris_names

        if info.source == 'primitive':
            return np.stack([self.pris[etype][names.index(c)]
                             for c in info.components], axis=-1)
        elif info.source == 'gradient':
            cols = []
            for c in info.components:
                var, _, d = c.rpartition('-')
                cols.append(self.grad_pris[etype][names.index(var)][int(d)])
            return np.stack(cols, axis=-1)
        elif info.source in ('aux', 'provider'):
            return self.fields.get((etype, info.name))
        else:
            raise ValueError(f'Unknown field source: {info.source!r}')

    def make_ploc_writable(self):
        # Copy-on-write: ensure self.ploc is independent of region.ploc before
        # a transformer mutates it.  After the sample is dropped, the copy is
        # gone and the next sample built from this region starts fresh.
        if self.ploc is self.region.ploc:
            self.ploc = {k: v.copy() for k, v in self.ploc.items()}

    @property
    def samples(self):
        # Tabular accessor for sampled conservative + (optional) grads on
        # PointsSnapshotRegion-built samples.  Layout: (npts, nfields) on root,
        # empty on non-root.  AttributeError elsewhere — only at_points()
        # regions populate the underlying _samps via their _compute_sample.
        samps = self._samps
        return samps.T if samps.size else samps

    def run(self, runner, *, public_only=True):
        # Run a FieldRunner against this sample.  Each provider gets a per-key
        # SampleView lens; reads pris/grad_pris/normals/state/etc., writes
        # into view.fields (collected back into self.fields).  Transformer
        # providers mutate ploc/pris in place via the view.  Callers build
        # the runner explicitly — cleaner than overloading on type, and the
        # runner can be reused (e.g. cached at plugin init, or re-queried
        # afterwards for .fields()).
        return runner.run_on_sample(self, public_only=public_only)

    def flat(self, name):
        # Unified all-points view that hides the etype split, for scripting:
        #   sample.flat('ploc')   -> (ndims, N)
        #   sample.flat('pris')   -> (nvars, N)
        #   sample.flat('mach')   -> (N,)
        etypes = self.region.etypes

        def cat(arrs, nlead):
            return np.concatenate(
                [a.reshape(*a.shape[:nlead], -1) for a in arrs], axis=-1)

        if name == 'ploc':
            return cat([self.ploc[et] for et in etypes], 1)
        elif name == 'pris':
            byet = self.pris
            nvars = len(byet[etypes[0]])
            return np.stack([cat([byet[et][i] for et in etypes], 0)
                             for i in range(nvars)])
        elif name == 'grad_pris':
            byet = self.grad_pris
            nvars = len(byet[etypes[0]])
            return np.stack([cat([byet[et][i] for et in etypes], 1)
                             for i in range(nvars)])
        else:
            return cat([self.fields[et, name] for et in etypes], 0)
