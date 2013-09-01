# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseFluidElements(object):
    _dynvarmap = {2: ['rho', 'u', 'v', 'p'],
                  3: ['rho', 'u', 'v', 'w', 'p']}

    def _process_ics(self, ics):
        rho, p = ics[0], ics[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in ics[1:-1]]

        # Compute the energy
        gamma = self._cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in ics[1:-1])

        return [rho] + rhovs + [E]


class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def get_tdisf_upts_kern(self):
        kc = self._cfg.items_as('constants', float)

        return self._be.kernel('tdisf_inv', self.ndims, self.nvars,
                               self.scal_upts_inb, self._smat_upts,
                               self._vect_upts[0], kc)
