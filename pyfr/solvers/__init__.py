# -*- coding: utf-8 -*-

from pyfr.integrators import get_integrator
from pyfr.solvers.base import BaseSystem
from pyfr.solvers.euler import EulerSystem
from pyfr.solvers.navstokes import NavierStokesSystem
from pyfr.util import subclass_map


def get_solver(backend, rallocs, mesh, initsoln, cfg):
    systemmap = subclass_map(BaseSystem, 'name')
    systemcls = systemmap[cfg.get('solver', 'system')]

    # Combine with an integrator to yield the solver
    return get_integrator(backend, systemcls, rallocs, mesh, initsoln, cfg)
