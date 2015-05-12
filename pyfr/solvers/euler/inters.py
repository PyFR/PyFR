# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)


class EulerIntInters(BaseAdvectionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.euler.kernels.intcflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal0_lhs, ur=self._scal0_rhs,
            magnl=self._mag_pnorm_lhs, magnr=self._mag_pnorm_rhs,
            nl=self._norm_pnorm_lhs
        )


class EulerMPIInters(BaseAdvectionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.euler.kernels.mpicflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs, dims=[self.ninterfpts],
            ul=self._scal0_lhs, ur=self._scal0_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class EulerBaseBCInters(BaseAdvectionBCInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.euler.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type, stv=self.stv)

        if(self.stv):
            self.kernels['comm_flux'] = lambda: self._be.kernel(
                'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal0_lhs,
                magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
                ploc=self._ploc_lhs
            )

        else:
            self.kernels['comm_flux'] = lambda: self._be.kernel(
                'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal0_lhs,
                magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
            )


class EulerSupInflowBCInters(EulerBaseBCInters):
    type = 'sup-in-fa'
    stv = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tpl_c['rho'] = self.cfg.get(self.cfgsect, 'rho')
        self._tpl_c['p'] = self.cfg.get(self.cfgsect, 'p')
        for d in 'uvw'[:self.ndims]:
            self._tpl_c[d] = self.cfg.get(self.cfgsect, d)


class EulerSubInflowFrvBCInters(EulerBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'
    stv = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tpl_c['rho'] = self.cfg.get(self.cfgsect, 'rho')
        for d in 'uvw'[:self.ndims]:
            self._tpl_c[d] = self.cfg.get(self.cfgsect, d)


class EulerCharRiemInvBCInters(EulerBaseBCInters):
    type = 'char-riem-inv'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tpl_c['p'], self._tpl_c['rho'] = self._eval_opts(['p', 'rho'])
        self._tpl_c['v'] = self._eval_opts('uvw'[:self.ndims])


class EulerSlpAdiaWallBCInters(EulerBaseBCInters):
    type = 'slp-adia-wall'
