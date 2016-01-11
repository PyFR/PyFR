# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import NullComputeKernel, NullMPIKernel
from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)


class NavierStokesIntInters(BaseAdvectionDiffusionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, rhs, elemap, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       visc_corr=visc_corr, c=self._tpl_c)

        # Generate the additional view matrices for artificial viscosity
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'artificial-viscosity':
            avis0_lhs = self._avis_view(lhs, 'get_avis_fpts_for_inter')
            avis0_rhs = self._avis_view(rhs, 'get_avis_fpts_for_inter')
            tplargs['art_vis'] = 'mu'
        else:
            avis0_lhs = avis0_rhs = None
            tplargs['art_vis'] = 'none'

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal0_lhs, urin=self._scal0_rhs,
            ulout=self._vect0_lhs, urout=self._vect0_rhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal0_lhs, ur=self._scal0_rhs,
            gradul=self._vect0_lhs, gradur=self._vect0_rhs,
            amul=avis0_lhs, amur=avis0_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class NavierStokesMPIInters(BaseAdvectionDiffusionMPIInters):
    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        super().__init__(be, lhs, rhsrank, rallocs, elemap, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       visc_corr=visc_corr, c=self._tpl_c)

        # Generate the additional kernels/views for artificial viscosity
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'artificial-viscosity':
            avis0_lhs = self._avis_xchg_view(lhs, 'get_avis_fpts_for_inter')
            avis0_rhs = be.xchg_matrix_for_view(avis0_lhs)

            # If we need to send our artificial viscosity to the RHS
            if self._tpl_c['ldg-beta'] != -0.5:
                self.kernels['avis_fpts_pack'] = lambda: be.kernel(
                    'pack', avis0_lhs
                )
                self.kernels['avis_fpts_send'] = lambda: be.kernel(
                    'send_pack', avis0_lhs, self._rhsrank, self.MPI_TAG
                )
            else:
                self.kernels['avis_fpts_pack'] = lambda: NullComputeKernel()
                self.kernels['avis_fpts_send'] = lambda: NullMPIKernel()

            # If we need to recv artificial viscosity from the RHS
            if self._tpl_c['ldg-beta'] != 0.5:
                self.kernels['avis_fpts_recv'] = lambda: be.kernel(
                    'recv_pack', avis0_rhs, self._rhsrank, self.MPI_TAG
                )
                self.kernels['avis_fpts_unpack'] = lambda: be.kernel(
                    'unpack', avis0_rhs
                )
            else:
                self.kernels['avis_fpts_recv'] = lambda: NullMPIKernel()
                self.kernels['avis_fpts_unpack'] = lambda: NullComputeKernel()

            tplargs['art_vis'] = 'mu'
        else:
            avis0_lhs = avis0_rhs = None
            tplargs['art_vis'] = 'none'

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'mpiconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal0_lhs, urin=self._scal0_rhs, ulout=self._vect0_lhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal0_lhs, ur=self._scal0_rhs,
            gradul=self._vect0_lhs, gradur=self._vect0_rhs,
            amul=avis0_lhs, amur=avis0_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class NavierStokesBaseBCInters(BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       visc_corr=visc_corr, c=self._tpl_c, bctype=self.type,
                       bccfluxstate=self.cflux_state)

        # Generate the additional view matrices for artificial viscosity
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'artificial-viscosity':
            avis0_lhs = self._avis_view(lhs, 'get_avis_fpts_for_inter')
            tplargs['art_vis'] = 'mu'
        else:
            avis0_lhs = None
            tplargs['art_vis'] = 'none'

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'bcconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal0_lhs, ulout=self._vect0_lhs,
            nlin=self._norm_pnorm_lhs, ploc=self._ploc
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal0_lhs, gradul=self._vect0_lhs, amul=avis0_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=self._ploc
        )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tpl_c['cpTw'], = self._eval_opts(['cpTw'])
        self._tpl_c['v'] = self._eval_opts('uvw'[:self.ndims], default='0')


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

        tplc, self._ploc = self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )

        self._tpl_c.update(tplc)


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc, self._ploc = self._exp_opts(
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

        tplc, self._ploc = self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u':0, 'v': 0, 'w':0}
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

        tplc, self._ploc = self._exp_opts(['p'], lhs)
        self._tpl_c.update(tplc)

