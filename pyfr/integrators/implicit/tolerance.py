from pyfr.util import subclass_where


def get_krylov_tol_controller(cfg, serialiser, initsoln):
    sect = 'solver-time-integrator'
    name = cfg.get(sect, 'krylov-tol-controller', 'none')
    cls = subclass_where(BaseToleranceController, name=name)

    return cls(cfg, serialiser, initsoln)


class BaseToleranceController:
    name = None
    max_retries = 0
    settled = True

    def update(self, wall_time, gamma_dt, success):
        pass

    def reset(self):
        pass

    def soft_reset(self):
        pass


class NullToleranceController(BaseToleranceController):
    name = 'none'

    def __init__(self, cfg, serialiser, initsoln):
        sect = 'solver-time-integrator'
        self._tol = cfg.getfloat(sect, 'krylov-rtol', 1e-2)

    def select_tolerance(self):
        return self._tol
