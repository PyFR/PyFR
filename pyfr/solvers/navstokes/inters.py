# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)


class NavierStokesIntInters(BaseAdvectionDiffusionIntInters):
    def get_comm_flux_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants

        return self._be.kernel('rsolve_ldg_vis_int', self.ndims, self.nvars,
                               rsinv, self._scal0_lhs, self._vect0_lhs,
                               self._scal0_rhs, self._vect0_rhs,
                               self._mag_pnorm_lhs, self._mag_pnorm_rhs,
                               self._norm_pnorm_lhs, kc)


class NavierStokesMPIInters(BaseAdvectionDiffusionMPIInters):
    def get_comm_flux_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants

        return self._be.kernel('rsolve_ldg_vis_mpi', self.ndims, self.nvars,
                               rsinv, self._scal0_lhs, self._vect0_lhs,
                               self._scal0_rhs, self._vect0_rhs,
                               self._mag_pnorm_lhs, self._norm_pnorm_lhs, kc)


class NavierStokesBaseBCInters(BaseAdvectionDiffusionBCInters):
    def get_comm_flux_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants

        return self._be.kernel('rsolve_ldg_vis_bc', self.ndims, self.nvars,
                               rsinv, self.type, self._scal0_lhs,
                               self._vect0_lhs, self._mag_pnorm_lhs,
                               self._norm_pnorm_lhs, kc)


class NavierStokesIsoThermNoslipBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-iso-wall'
    args = ['cpTw']


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    args = ['rho', 'p', 'u', 'v', 'w']


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'


class NavierStokesSubInflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    args = ['rho', 'u', 'v', 'w']


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    args = ['p']
