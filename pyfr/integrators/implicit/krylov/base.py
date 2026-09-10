from pyfr.integrators.implicit.base import BaseImplicitIntegrator


class BaseLinearSolver(BaseImplicitIntegrator):
    linear_name = None

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-time-integrator'

        # Matrix-vector product budget per linear solve
        self._krylov_nmax = cfg.getint(sect, 'linear-max-iter', 10)

        # Threshold below which Krylov quantities are treated as breakdown
        self._breakdown_tol = 1e3*backend.fpdtype_eps

        super().__init__(backend, systemcls, mesh, initsoln, cfg)
