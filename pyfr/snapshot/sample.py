from pyfr.util import subclass_where


class BaseSampleView:
    layout = None

    def __init__(self, sample, etype):
        self.sample = sample
        self.etype = etype
        self.cfg = sample.region.cfg
        self.fields = {}

    @property
    def pris(self):
        return self.sample.pris[self.etype]

    @property
    def grad_pris(self):
        return self.sample.grad_pris[self.etype]

    @property
    def normals(self):
        return getattr(self.sample.region, 'normals', {}).get(self.etype)

    @property
    def min_upt_wall_dist_approx(self):
        return getattr(self.sample.region, 'wall_dist', {}).get(self.etype)

    @property
    def has_grads(self):
        return self.grad_pris is not None


class SoASampleView(BaseSampleView):
    layout = 'soa'

    @property
    def ploc(self):
        return self.sample.ploc[self.etype]

    def field_array(self, info):
        arr = self.sample.field_array(self.etype, info)
        return arr.T if arr is not None else None


class AoSSampleView(BaseSampleView):
    layout = 'aos'

    @property
    def ploc(self):
        return self.sample.ploc[self.etype].T

    def field_array(self, info):
        return self.sample.field_array(self.etype, info)


class SnapshotSample:
    def __init__(self, region, snap):
        self.region = region
        self.snap = snap

        self.field_arrays = {}
        self.fields = {}

        self.ploc = {k: v.copy() for k, v in region.ploc.items()}

        region._compute_sample(self, snap)
        self._register_fields(snap)

    def view(self, etype, *, layout='soa'):
        return subclass_where(BaseSampleView, layout=layout)(self, etype)

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

    @property
    def samples(self):
        samps = self._samps
        return samps.T if samps.size else samps

    def run(self, runner, *, public_only=True):
        return runner.run_on_sample(self, public_only=public_only)
