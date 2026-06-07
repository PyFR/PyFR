from functools import cached_property
from math import prod

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.snapshot.fieldinfo import FieldInfo
from pyfr.snapshot.region import PointsSnapshotRegion
from pyfr.util import subclass_where


class BaseSnapshot:
    data = None
    grad_data = {}
    prefix = None
    state = None

    _data_field_names = ()

    @property
    def ndims(self):
        return self.mesh.ndims

    def at_points(self, ppts):
        return PointsSnapshotRegion(self.mesh, self.cfg, ppts)

    def aux(self, etype):
        return {}

    def aux_info(self, etype):
        return {}

    def to_pris(self, interp_data):
        pass

    def to_grad_pris(self, interp_data, grad_interp):
        pass

    @cached_property
    def fields(self):
        out = {}
        dtype = np.dtype(self.dtype)

        for i, name in enumerate(self._data_field_names):
            out[name] = FieldInfo(name=name, kind='point',
                                  dtype=dtype, source='data',
                                  components=(name,), data_index=i)

        if self.has_grads:
            for i, name in enumerate(self._data_field_names):
                gname = f'grad {name}'
                components = tuple(f'{name}-{d}' for d in range(self.ndims))
                out[gname] = FieldInfo(
                    name=gname, kind='point', dtype=dtype,
                    source='grad_data', components=components, data_index=i
                )

        self._append_aux_fields(out)
        return out

    def iter_fields(self):
        dtype = np.dtype(self.dtype)

        for name, varnames in self.primitive_var_groups().items():
            yield FieldInfo(name=name, kind='point', dtype=dtype,
                            source='primitive',
                            components=varnames)

        if self.has_grads:
            for name, varnames in self.primitive_var_groups().items():
                gname = f'grad {name}'
                gcomps = tuple(f'{c}-{d}' for c in varnames
                               for d in range(self.ndims))
                yield FieldInfo(name=gname, kind='point', dtype=dtype,
                                source='gradient', components=gcomps)

        seen = set()
        pshapes = self._aux_pshapes()
        for et in self.ele_types:
            for name, (per_ele, adtype) in self.aux_info(et).items():
                if name in seen:
                    continue
                seen.add(name)
                yield self._aux_field_info(name, per_ele, adtype, pshapes)

    def primitive_at(self, etype, info, sample):
        names = self.pris_names
        return np.stack([sample.pris[etype][names.index(c)]
                         for c in info.components], axis=-1)

    def gradient_at(self, etype, info, sample):
        names = self.pris_names
        cols = []
        for c in info.components:
            var, _, d = c.rpartition('-')
            cols.append(sample.grad_pris[etype][names.index(var)][int(d)])
        return np.stack(cols, axis=-1)

    def data_at(self, etype, info):
        # data[etype][:, info.data_index] is (nsvpts, neles); canonical AoS
        # is (flat_npts, 1) with element-major flat order.
        arr = self.data[etype][:, info.data_index]
        return arr.T.reshape(-1, 1)

    def grad_data_at(self, etype, info):
        # grad_data[etype][:, :, info.data_index] is (ndims, nsvpts, neles);
        # canonical AoS is (flat_npts, ndims), element-major.
        arr = self.grad_data[etype][:, :, info.data_index]
        return arr.transpose(2, 1, 0).reshape(-1, arr.shape[0])

    def _aux_pshapes(self):
        if not self.ele_types:
            return set()
        et0 = self.ele_types[0]
        shapecls = subclass_where(BaseShape, name=et0)
        sh = shapecls(self.mesh.spts[et0].shape[0], self.cfg)
        return {(sh.nupts,), (len(sh.linspts),)}

    def _aux_field_info(self, name, per_ele, dtype, pshapes):
        if per_ele in pshapes:
            return FieldInfo(name=name, kind='point',
                             dtype=dtype, source='aux', components=(name,))
        if len(per_ele) > 1 and per_ele[:-1] in pshapes:
            components = tuple(f'{name}-{d}' for d in range(per_ele[-1]))
            return FieldInfo(
                name=name, kind='point', dtype=dtype,
                source='aux', components=components
            )
        ncomps = prod(per_ele)
        components = tuple(f'{name}-{i}' for i in range(ncomps))
        return FieldInfo(name=name, kind='cell',
                         dtype=dtype, source='aux', components=components)

    def _append_aux_fields(self, out):
        pshapes = self._aux_pshapes()
        for et in self.ele_types:
            for name, (per_ele, dtype) in self.aux_info(et).items():
                if name in out:
                    continue
                out[name] = self._aux_field_info(name, per_ele, dtype,
                                                 pshapes)
