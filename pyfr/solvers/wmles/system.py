from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionSystem
from pyfr.solvers.wmles.elements import WMLESElements
from pyfr.solvers.wmles.inters import (WMLESIntInters,
                                       WMLESMPIInters,
                                       WMLESBaseBCInters)


class WMLESSystem(BaseAdvectionDiffusionSystem):
    name = 'wmles'

    elementscls = WMLESElements
    intinterscls = WMLESIntInters
    mpiinterscls = WMLESMPIInters
    bbcinterscls = WMLESBaseBCInters
