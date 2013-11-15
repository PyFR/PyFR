# -*- coding: utf-8 -*-

import numpy as np

from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)


class NavierStokesIntInters(BaseAdvectionDiffusionIntInters):
    def __init__(self, *args, **kwargs):
        super(NavierStokesIntInters, self).__init__(*args, **kwargs)

        # Pointwise template arguments
        rsolver = self._cfg.get('solver-interfaces', 'riemann-solver')
        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, c=self._tpl_c)

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
                             rsolver=rsolver, c=self._tpl_c)

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
                             rsolver=rsolver, c=self._tpl_c,
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

    def __init__(self, *args, **kwargs):
        super(NavierStokesNoSlpIsoWallBCInters, self).__init__(*args, **kwargs)

        self._tpl_c['cpTw'], = self._eval_opts(['cpTw'])
        self._tpl_c['v'] = self._eval_opts('uvw'[:self.ndims], default='0')


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'

    def __init__(self, *args, **kwargs):
        super(NavierStokesSupInflowBCInters, self).__init__(*args, **kwargs)

        self._tpl_c['rho'], self._tpl_c['p'] = self._eval_opts(['rho', 'p'])
        self._tpl_c['v'] = self._eval_opts('uvw'[:self.ndims])


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'

    def __init__(self, *args, **kwargs):
        super(NavierStokesSubInflowFrvBCInters, self).__init__(*args, **kwargs)

        self._tpl_c['rho'], = self._eval_opts(['rho'])
        self._tpl_c['v'] = self._eval_opts('uvw'[:self.ndims])


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'

    def __init__(self, *args, **kwargs):
        super(NavierStokesSubInflowFtpttangBCInters, self).__init__(*args,
                                                                    **kwargs)

        gamma = self._cfg.getfloat('constants', 'gamma')

        # Pass boundary constants to the backend
        self._tpl_c['cpTt'], = self._eval_opts(['cpTt'])
        self._tpl_c['pt'], = self._eval_opts(['pt'])
        self._tpl_c['Rdcp'] = (gamma - 1.0)/gamma

        # Calculate u, v velocity components from the inflow angle
        theta = self._eval_opts(['theta'])[0]*np.pi/180.0
        velcomps = np.array([np.cos(theta), np.sin(theta), 1.0])

        # Adjust u, v and calculate w velocity components for 3-D
        if self.ndims == 3:
            phi = self._eval_opts(['phi'])[0]*np.pi/180.0
            velcomps[:2] *= np.sin(phi)
            velcomps[2] *= np.cos(phi)

        self._tpl_c['vc'] = velcomps[:self.ndims]


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'

    def __init__(self, *args, **kwargs):
        super(NavierStokesSubOutflowBCInters, self).__init__(*args, **kwargs)

        self._tpl_c['p'], = self._eval_opts(['p'])
