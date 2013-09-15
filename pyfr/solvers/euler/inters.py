# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)


class EulerIntInters(BaseAdvectionIntInters):
    def get_comm_flux_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants

        return self._be.kernel('rsolve_inv_int', self.ndims, self.nvars,
                               rsinv, self._scal0_lhs, self._scal0_rhs,
                               self._mag_pnorm_lhs, self._mag_pnorm_rhs,
                               self._norm_pnorm_lhs, kc)


class EulerMPIInters(BaseAdvectionMPIInters):
    def get_comm_flux_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants

        return self._be.kernel('rsolve_inv_mpi', self.ndims, self.nvars,
                               rsinv, self._scal0_lhs, self._scal0_rhs,
                               self._mag_pnorm_lhs, self._norm_pnorm_lhs, kc)


class EulerBaseBCInters(BaseAdvectionBCInters):
    def get_comm_flux_kern(self):
        rsinv = self._cfg.get('solver-interfaces', 'riemann-solver')
        kc = self._kernel_constants

        return self._be.kernel('rsolve_inv_bc', self.ndims, self.nvars,
                               rsinv, self.type, self._scal0_lhs,
                               self._mag_pnorm_lhs, self._norm_pnorm_lhs, kc)
