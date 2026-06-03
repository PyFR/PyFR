class SnapshotSample:
    def __init__(self, region, snap):
        self.region = region
        self.snap = snap

        self.field_arrays = {}
        self.fields = {}

        self.ploc = region.ploc

        region._compute_sample(self, snap)
        self._register_fields(snap)

    def _register_fields(self, snap):
        for info in snap.iter_fields():
            self.fields[info.name] = info

    def field_array(self, etype, info):
        if (etype, info.name) in self.field_arrays:
            return self.field_arrays[etype, info.name]
        if info.source == 'primitive':
            return self.snap.primitive_at(etype, info, self)
        elif info.source == 'gradient':
            return self.snap.gradient_at(etype, info, self)
        elif info.source == 'data':
            return self.snap.data_at(etype, info)
        elif info.source == 'grad_data':
            return self.snap.grad_data_at(etype, info)
        elif info.source in ('aux', 'provider'):
            return None
        else:
            raise ValueError(f'Unknown field source: {info.source!r}')

    def make_ploc_writable(self):
        if self.ploc is self.region.ploc:
            self.ploc = {k: v.copy() for k, v in self.ploc.items()}

    @property
    def samples(self):
        samps = self._samps
        return samps.T if samps.size else samps

    def run(self, runner, *, public_only=True):
        return runner.run_on_sample(self, public_only=public_only)
