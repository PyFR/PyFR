import numpy as np

from pyfr.cache import memoize
from pyfr.plugins.fields import get_field_providers
from pyfr.snapshot import FieldInfo
from pyfr.snapshot.sample import SoASampleView


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
            view = SoASampleView(sample, et)
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
