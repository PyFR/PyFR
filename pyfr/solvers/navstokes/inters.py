# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base.kernels import ComputeMetaKernel
from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)


class NavierStokesIntInters(BaseAdvectionDiffusionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, rhs, elemap, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction')
        shock_capturing = self.cfg.get('solver', 'shock-capturing')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       visc_corr=visc_corr, shock_capturing=shock_capturing,
                       c=self._tpl_c)

        be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        if abs(self._tpl_c['ldg-beta']) == 0.5:
            self.kernels['copy_fpts'] = lambda: ComputeMetaKernel(
                [ele.kernels['_copy_fpts']() for ele in elemap.values()]
            )

        self.kernels['con_u'] = lambda: be.kernel(
            'intconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs,
            ulout=self._vect_lhs, urout=self._vect_rhs
        )
        self.kernels['comm_flux'] = lambda: be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class NavierStokesMPIInters(BaseAdvectionDiffusionMPIInters):
    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        super().__init__(be, lhs, rhsrank, rallocs, elemap, cfg)

        # Pointwise template arguments
        rsolver = cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = cfg.get('solver', 'viscosity-correction')
        shock_capturing = cfg.get('solver', 'shock-capturing')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       visc_corr=visc_corr, shock_capturing=shock_capturing,
                       c=self._tpl_c)

        be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: be.kernel(
            'mpiconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs, ulout=self._vect_lhs
        )
        self.kernels['comm_flux'] = lambda: be.kernel(
            'mpicflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class NavierStokesBaseBCInters(BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        # Pointwise template arguments
        rsolver = cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = cfg.get('solver', 'shock-capturing', 'none')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       visc_corr=visc_corr, shock_capturing=shock_capturing,
                       c=self._tpl_c, bctype=self.type,
                       bccfluxstate=self.cflux_state)

        be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

        self.kernels['con_u'] = lambda: be.kernel(
            'bcconu', tplargs=tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ulin=self._scal_lhs,
            ulout=self._vect_lhs, nlin=self._norm_pnorm_lhs,
            **self._external_vals
        )
        self.kernels['comm_flux'] = lambda: be.kernel(
            'bccflux', tplargs=tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self._scal_lhs,
            gradul=self._vect_lhs, magnl=self._mag_pnorm_lhs,
            nl=self._norm_pnorm_lhs, artviscl=self._artvisc_lhs,
            **self._external_vals
        )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self._tpl_c['cpTw'], = self._eval_opts(['cpTw'])
        self._tpl_c.update(
            self._exp_opts('uvw'[:self.ndims], lhs,
                           default={'u': 0, 'v': 0, 'w': 0})
        )


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost'


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc = self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )
        self._tpl_c.update(tplc)


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc = self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )
        self._tpl_c.update(tplc)


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc = self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )
        self._tpl_c.update(tplc)


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        gamma = self.cfg.getfloat('constants', 'gamma')

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
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self._tpl_c.update(self._exp_opts(['p'], lhs))
