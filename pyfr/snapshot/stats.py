from functools import cached_property

import numpy as np

from pyfr.snapshot.base import FieldInfo
from pyfr.snapshot.file import FileSnapshot


class StatsSnapshot(FileSnapshot):
    # Stored-form output (tavg, residual, ...).  The file's stored field
    # names ARE the primitives; no con_to_pri transform is needed.  No
    # gradients in the file format.  No providers (they assume conservative
    # form).  Field registry is a flat list of 1-component scalars.
    name = 'stats'
    has_grads = False
    supports_providers = False

    @property
    def pris_names(self):
        return list(self.stored_fields)

    def to_pris(self, interp_data, cfg):
        # Identity: stored data IS the primitive layout.  list() to match
        # the contract (one array per pris_names entry).
        return list(interp_data)

    def to_grad_pris(self, interp_data, grad_interp, cfg):
        # Stored-form files don't carry gradients.
        return None

    @cached_property
    def fields(self):
        # Flat 1-component scalars named after their stored field, plus aux.
        out = {}
        dtype = np.dtype(self.dtype)

        for name in self.pris_names:
            out[name] = FieldInfo(name=name, kind='point', ncomps=1,
                                  dtype=dtype, source='primitive',
                                  components=(name,))

        self._append_aux_fields(out)
        return out
