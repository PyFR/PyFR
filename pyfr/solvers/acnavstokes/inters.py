from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)


class ACNavierStokesIntInters(BaseAdvectionDiffusionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, rhs, elemap, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self.c)

        kprefix = 'pyfr.solvers.acnavstokes.kernels'
        self._be.pointwise.register(f'{kprefix}.intconu')
        self._be.pointwise.register(f'{kprefix}.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs,
            ulout=self._comm_lhs, urout=self._comm_rhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            nl=self._pnorm_lhs
        )


class ACNavierStokesMPIInters(BaseAdvectionDiffusionMPIInters):
    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super().__init__(be, lhs, rhsrank, elemap, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self.c)

        kprefix = 'pyfr.solvers.acnavstokes.kernels'
        self._be.pointwise.register(f'{kprefix}.mpiconu')
        self._be.pointwise.register(f'{kprefix}.mpicflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'mpiconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs, ulout=self._comm_lhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            nl=self._pnorm_lhs
        )


class ACNavierStokesBaseBCInters(BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self.c, bctype=self.type,
                       bccfluxstate=self.cflux_state)

        self._be.pointwise.register('pyfr.solvers.acnavstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.acnavstokes.kernels.bccflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'bcconu', tplargs=tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ulin=self._scal_lhs,
            ulout=self._comm_lhs, nlin=self._pnorm_lhs, **self._external_vals
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self._scal_lhs,
            gradul=self._vect_lhs, nl=self._pnorm_lhs, **self._external_vals
        )


class ACNavierStokesNoSlpWallBCInters(ACNavierStokesBaseBCInters):
    type = 'no-slp-wall'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})


class ACNavierStokesSlpWallBCInters(ACNavierStokesBaseBCInters):
    type = 'slp-wall'
    cflux_state = None


class ACNavierStokesInflowBCInters(ACNavierStokesBaseBCInters):
    type = 'ac-in-fv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts('uvw'[:self.ndims], lhs)


class ACNavierStokesOutflowBCInters(ACNavierStokesBaseBCInters):
    type = 'ac-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts('p', lhs)


class ACNavierStokesCharRiemInvBCInters(ACNavierStokesBaseBCInters):
    type = 'ac-char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c['niters'] = cfg.getint(cfgsect, 'niters', 4)
        self.c['bc-ac-zeta'] = cfg.getfloat(cfgsect, 'ac-zeta')
        self.c |= self._exp_opts(
            ['p', 'u', 'v', 'w'][:self.ndims + 1], lhs
        )
