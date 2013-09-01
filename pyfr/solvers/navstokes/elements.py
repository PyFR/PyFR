# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    def get_tdisf_upts_kern(self):
        kc = self._cfg.items_as('constants', float)

        return self._be.kernel('tdisf_vis', self.ndims, self.nvars,
                               self.scal_upts_inb, self._smat_upts,
                               self._rcpdjac_upts, self._vect_upts[0], kc)
