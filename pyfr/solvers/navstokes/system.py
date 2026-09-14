from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionSystem
from pyfr.solvers.navstokes.elements import NavierStokesElements
from pyfr.solvers.navstokes.inters import (NavierStokesBaseBCInters,
                                           NavierStokesIntInters,
                                           NavierStokesMPIInters,
                                           NavierStokesPeriodicInters)


class NavierStokesSystem(BaseAdvectionDiffusionSystem):
    name = 'navier-stokes'
    ef_solver = 'euler'

    elementscls = NavierStokesElements
    intinterscls = NavierStokesIntInters
    mpiinterscls = NavierStokesMPIInters
    bbcinterscls = NavierStokesBaseBCInters
    pinterscls = NavierStokesPeriodicInters
