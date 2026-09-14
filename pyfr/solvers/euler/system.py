from pyfr.solvers.baseadvec import BaseAdvectionSystem
from pyfr.solvers.euler.elements import EulerElements
from pyfr.solvers.euler.inters import (EulerIntInters, EulerMPIInters,
                                       EulerBaseBCInters,
                                       EulerPeriodicInters)


class EulerSystem(BaseAdvectionSystem):
    name = 'euler'
    ef_solver = 'euler'

    elementscls = EulerElements
    intinterscls = EulerIntInters
    mpiinterscls = EulerMPIInters
    bbcinterscls = EulerBaseBCInters
    pinterscls = EulerPeriodicInters
