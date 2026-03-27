
from pyfr.integrators.implicit.base import BaseImplicitIntegrator


class BaseKrylovSolver(BaseImplicitIntegrator):
    krylov_name = None

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-time-integrator'

        # Finite difference perturbation for JFNK
        if cfg.hasopt(sect, 'krylov-eps'):
            self._krylov_eps = cfg.getfloat(sect, 'krylov-eps')
        else:
            self._krylov_eps = backend.fpdtype_eps**0.5

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        # Precision-dependent breakdown tolerance
        self._breakdown_tol = 1e3*self.backend.fpdtype_eps
