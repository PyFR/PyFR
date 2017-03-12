# -*- coding: utf-8 -*-

from pyfr.solvers.acnavstokes.elements import ACNavierStokesElements
from pyfr.solvers.acnavstokes.inters import (ACNavierStokesBaseBCInters,
                                             ACNavierStokesIntInters,
                                             ACNavierStokesMPIInters)
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionSystem


class ACNavierStokesSystem(BaseAdvectionDiffusionSystem):
    name = 'ac-navier-stokes'

    elementscls = ACNavierStokesElements
    intinterscls = ACNavierStokesIntInters
    mpiinterscls = ACNavierStokesMPIInters
    bbcinterscls = ACNavierStokesBaseBCInters
