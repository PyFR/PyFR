# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem
from pyfr.solvers.euler.elements import EulerElements
from pyfr.solvers.euler.inters import (EulerIntInters, EulerMPIInters,
                                       EulerBaseBCInters)


class EulerSystem(BaseAdvectionSystem):
    name = 'euler'

    elementscls = EulerElements
    intinterscls = EulerIntInters
    mpiinterscls = EulerMPIInters
    bbcinterscls = EulerBaseBCInters
