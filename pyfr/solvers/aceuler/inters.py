from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)


class ACEulerIntInters(BaseAdvectionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.aceuler.kernels.intcflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self.c)

        self._set_external('ac_zeta', 'scalar fpdtype_t')
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs, nl=self._pnorm_lhs,
            extrns=self._external_args, 
        )


class ACEulerMPIInters(BaseAdvectionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.aceuler.kernels.mpicflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self.c)

        self._set_external('ac_zeta', 'scalar fpdtype_t')
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs, 
            nl=self._pnorm_lhs, extrns=self._external_args, 
        )


class ACEulerBaseBCInters(BaseAdvectionBCInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.aceuler.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self.c, bctype=self.type)

        self._set_external('ac_zeta', 'scalar fpdtype_t')
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self._scal_lhs, nl=self._pnorm_lhs,
        )


class ACEulerInflowBCInters(ACEulerBaseBCInters):
    type = 'ac-in-fv'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts('uvw'[:self.ndims], lhs)


class ACEulerOutflowBCInters(ACEulerBaseBCInters):
    type = 'ac-out-fp'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts('p', lhs)


class ACEulerSlpWallBCInters(ACEulerBaseBCInters):
    type = 'slp-wall'


class ACEulerCharRiemInvBCInters(ACEulerBaseBCInters):
    type = 'ac-char-riem-inv'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c['niters'] = cfg.getint(cfgsect, 'niters', 4)
        self.c['bc-ac-zeta'] = cfg.getfloat(cfgsect, 'ac-zeta')
        self.c |= self._exp_opts(
            ['p', 'u', 'v', 'w'][:self.ndims + 1], lhs
        )
