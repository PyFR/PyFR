import numpy as np

from pyfr.cache import memoize
from pyfr.plugins.fields import get_field_providers
from pyfr.snapshot import FieldInfo


class SampleView:
    def __init__(self, sample, etype, *, layout='soa'):
        self.sample = sample
        self.etype = etype
        self.layout = layout
        self.cfg = sample.region.cfg
        self.fields = {}

    @property
    def ploc(self):
        p = self.sample.ploc[self.etype]
        return p.T if self.layout == 'aos' else p

    @property
    def pris(self):
        return self.sample.pris[self.etype]

    @property
    def grad_pris(self):
        return self.sample.grad_pris[self.etype]

    @property
    def normals(self):
        region = self.sample.region
        return getattr(region, 'normals', {}).get(self.etype)

    @property
    def min_upt_wall_dist_approx(self):
        region = self.sample.region
        return getattr(region, 'wall_dist', {}).get(self.etype)

    @property
    def has_grads(self):
        return self.grad_pris is not None

    def field_array(self, info):
        arr = self.sample.field_array(self.etype, info)
        if arr is None:
            return None
        return arr.T if self.layout == 'soa' else arr


class FieldRunner:
    def __init__(self, names, ndims, cfg, export_type):
        self.ndims = ndims
        self.plugins = get_field_providers(names, ndims, cfg, export_type)

    @memoize
    def fields(self, public_only=False):
        out = {}
        for p in self.plugins:
            for fname, varnames in p.fields.items():
                if public_only and fname.startswith('_'):
                    continue

                out[fname] = varnames

        return out

    def run_on_sample(self, sample, public_only=False):
        clean = sample.region.clean
        for et in sample.region.etypes:
            view = SampleView(sample, et)
            for p in self.plugins:
                p.run(view)
            for fname, arr in view.fields.items():
                if public_only and fname.startswith('_'):
                    continue
                # Canonical AoS: (flat_npts, ncomps), element-major for raw
                if clean:
                    arr = arr[:, None] if arr.ndim == 1 else arr
                elif arr.ndim == 2:
                    arr = arr.T.reshape(-1, 1)
                else:
                    arr = arr.swapaxes(0, 1).reshape(-1, arr.shape[-1])
                sample.field_arrays[et, fname] = arr

        for p in self.plugins:
            for fname, varnames in p.fields.items():
                private = public_only and fname.startswith('_')
                if private or fname in sample.fields:
                    continue
                sample.fields[fname] = FieldInfo(
                    name=fname, kind='point',
                    dtype=np.dtype(sample.snap.dtype), source='provider',
                    components=varnames)

        return sample.field_arrays

    @property
    def needs_grads(self):
        return any(p.needs_grads for p in self.plugins)

    def __bool__(self):
        return bool(self.plugins)
