# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)


class NavierStokesIntInters(BaseAdvectionDiffusionIntInters):
    def __init__(self, *args, **kwargs):
        super(NavierStokesIntInters, self).__init__(*args, **kwargs)

        # Pointwise template arguments
        rsolver = self._cfg.get('solver-interfaces', 'riemann-solver')
        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, c=self._kernel_constants)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

    def get_con_u_kern(self):
        return self._be.kernel('intconu', self._tplargs,
                               dims=[self.ninterfpts],
                               ulin=self._scal0_lhs, urin=self._scal0_rhs,
                               ulout=self._scal1_lhs, urout=self._scal1_rhs)

    def get_comm_flux_kern(self):
        return self._be.kernel('intcflux', self._tplargs,
                               dims=[self.ninterfpts],
                               ul=self._scal0_lhs, ur=self._scal0_rhs,
                               gradul=self._vect0_lhs, gradur=self._vect0_rhs,
                               magnl=self._mag_pnorm_lhs,
                               magnr=self._mag_pnorm_rhs,
                               nl=self._norm_pnorm_lhs)


class NavierStokesMPIInters(BaseAdvectionDiffusionMPIInters):
    def __init__(self, *args, **kwargs):
        super(NavierStokesMPIInters, self).__init__(*args, **kwargs)

        # Pointwise template arguments
        rsolver = self._cfg.get('solver-interfaces', 'riemann-solver')
        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, c=self._kernel_constants)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

    def get_con_u_kern(self):
        return self._be.kernel('mpiconu', self._tplargs,
                               dims=[self.ninterfpts],
                               ulin=self._scal0_lhs, urin=self._scal0_rhs,
                               ulout=self._scal1_lhs)

    def get_comm_flux_kern(self):
        return self._be.kernel('mpicflux', self._tplargs,
                               dims=[self.ninterfpts],
                               ul=self._scal0_lhs, ur=self._scal0_rhs,
                               gradul=self._vect0_lhs, gradur=self._vect0_rhs,
                               magnl=self._mag_pnorm_lhs,
                               nl=self._norm_pnorm_lhs)


class NavierStokesBaseBCInters(BaseAdvectionDiffusionBCInters):
    def __init__(self, *args, **kwargs):
        super(NavierStokesBaseBCInters, self).__init__(*args, **kwargs)

        # Pointwise template arguments
        rsolver = self._cfg.get('solver-interfaces', 'riemann-solver')
        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, c=self._kernel_constants,
                             bctype=self.type)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

    def get_con_u_kern(self):
        return self._be.kernel('bcconu', self._tplargs, dims=[self.ninterfpts],
                               ulin=self._scal0_lhs, ulout=self._scal1_lhs)

    def get_comm_flux_kern(self):
        return self._be.kernel('bccflux', self._tplargs,
                               dims=[self.ninterfpts],
                               ul=self._scal0_lhs, gradul=self._vect0_lhs,
                               magnl=self._mag_pnorm_lhs,
                               nl=self._norm_pnorm_lhs)


class NavierStokesNoSlpIsoWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-iso-wall'

    @property
    def _kernel_constants(self):
        kc = super(NavierStokesNoSlpIsoWallBCInters, self)._kernel_constants
        kc = dict(kc)

        kc['cpTw'], = self._eval_opts('cpTw')

        return kc


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'

    @property
    def _kernel_constants(self):
        kc = super(NavierStokesSupInflowBCInters, self)._kernel_constants
        kc = dict(kc)

        kc['rho'], kc['p'] = self._eval_opts('rho', 'p')
        kc['v'] = self._eval_opts(*'uvw'[:self.ndims])

        return kc


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'


class NavierStokesSubInflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'

    @property
    def _kernel_constants(self):
        kc = super(NavierStokesSubInflowBCInters, self)._kernel_constants
        kc = dict(kc)

        kc['rho'], = self._eval_opts('rho')
        kc['v'] = self._eval_opts(*'uvw'[:self.ndims])

        return kc

class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'

    @property
    def _kernel_constants(self):
        kc = super(NavierStokesSubOutflowBCInters, self)._kernel_constants
        kc = dict(kc)

        kc['p'], = self._eval_opts('p')

        return kc
