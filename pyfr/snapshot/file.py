import numpy as np

from pyfr.snapshot.base import BaseSnapshot
from pyfr.util import subclass_where


class FileSnapshot(BaseSnapshot):
    def __init__(self, mesh, soln):
        from pyfr.solvers.base import BaseSystem

        self._soln = soln

        cfg = soln.config
        stats = soln.stats
        syscls = subclass_where(BaseSystem,
                                name=cfg.get('solver', 'system'))

        self.mesh = mesh
        self.cfg = cfg
        self.stats = stats
        self.elementscls = syscls.elementscls
        self.ele_types = list(soln.data)
        if soln.data:
            self.dtype = next(iter(soln.data.values())).dtype.type
        else:
            prec = cfg.get('backend', 'precision', 'single')
            self.dtype = np.float32 if prec == 'single' else np.float64
        self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
        self.cycle = stats.getint('solver-time-integrator', 'nacptsteps', 0)
        self.has_grads = bool(soln.grad_data)

        self._data_field_names = list(soln.fields)

        self.data = soln.data
        self.grad_data = soln.grad_data or {}
        self.prefix = stats.get('data', 'prefix')
        self.state = soln.state or {}

    def aux(self, etype):
        return self._soln.aux.get(etype, {})

    def aux_info(self, etype):
        return {name: (arr.shape[1:], arr.dtype)
                for name, arr in self._soln.aux.get(etype, {}).items()}
