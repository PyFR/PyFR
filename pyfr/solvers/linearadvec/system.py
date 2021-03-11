# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem
from pyfr.solvers.linearadvec.elements import LinearAdvecElements
from pyfr.solvers.linearadvec.inters import (LinearAdvecIntInters, LinearAdvecMPIInters,
                                             LinearAdvecBaseBCInters)


class LinearAdvecSystem(BaseAdvectionSystem):
    name = 'linearadvec'

    elementscls = LinearAdvecElements
    intinterscls = LinearAdvecIntInters
    mpiinterscls = LinearAdvecMPIInters
    bbcinterscls = LinearAdvecBaseBCInters
