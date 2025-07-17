from pyfr.mpiutil import mpi
from pyfr.plugins.base import init_csv
from pyfr.quadrules import SurfaceMixin
from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)

import numpy as np


class FluidIntIntersMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._ef_enabled:
            self._be.pointwise.register('pyfr.solvers.euler.kernels.intcent')

            self.kernels['comm_entropy'] = lambda: self._be.kernel(
                'intcent', tplargs={}, dims=[self.ninters],
                entmin_lhs=self._entmin_lhs, entmin_rhs=self._entmin_rhs
            )


class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        if self.cfg.get('solver', 'shock-capturing') == 'entropy-filter':
            self.p_min = self.cfg.getfloat('solver-entropy-filter', 'p-min',
                                           1e-6)
        else:
            self.p_min = self.cfg.getfloat('solver-interfaces', 'p-min',
                                           5*self._be.fpdtype_eps)

        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, c=self.c, p_min=self.p_min)


class FluidMPIIntersMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._ef_enabled:
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mpicent')

            self.kernels['comm_entropy'] = lambda: self._be.kernel(
                'mpicent', tplargs={}, dims=[self.ninters],
                entmin_lhs=self._entmin_lhs, entmin_rhs=self._entmin_rhs
            )


class EulerIntInters(TplargsMixin, FluidIntIntersMixin,
                     BaseAdvectionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.euler.kernels.intcflux')

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs, nl=self._pnorm_lhs
        )


class EulerMPIInters(TplargsMixin, FluidMPIIntersMixin,
                     BaseAdvectionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.euler.kernels.mpicflux')

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', self._tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs, nl=self._pnorm_lhs
        )


class EulerBaseBCInters(TplargsMixin, BaseAdvectionBCInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.euler.kernels.bccflux')

        self._tplargs |= dict(bctype=self.type, ninters=self.ninters)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self._scal_lhs, nl=self._pnorm_lhs,
            **self._external_vals
        )

        if self._ef_enabled:
            self._be.pointwise.register('pyfr.solvers.euler.kernels.bccent')

            self.kernels['comm_entropy'] = lambda: self._be.kernel(
                'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
                extrns=self._external_args, entmin_lhs=self._entmin_lhs,
                nl=self._pnorm_lhs, ul=self._scal_lhs, **self._external_vals
            )


class EulerSupInflowBCInters(EulerBaseBCInters):
    type = 'sup-in-fa'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class EulerSupOutflowBCInters(EulerBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class EulerCharRiemInvBCInters(EulerBaseBCInters):
    type = 'char-riem-inv'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class EulerSlpAdiaWallBCInters(EulerBaseBCInters):
    type = 'slp-adia-wall'


class MassFlowBCMixin(SurfaceMixin):
    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs
        )

        self.target_mfr = self.cfg.getfloat(cfgsect, 'mass-flow-rate')
        # When to start the mass flow controller
        self.tstart = self.cfg.getfloat(cfgsect, 'tstart', 0.0)
        # Start p value
        self.p = self.cfg.getfloat(cfgsect, 'p')
        self.bcname = cfgsect.removeprefix('soln-bcs-')
        # Mass flow average
        self.mf_avg = 0.0
        self.mf_alpha = 0.4
        # Parameter to control the strength of the controller
        self.eta = self.cfg.getfloat(cfgsect, 'eta')
        # Frequency that mf.csv should be updated
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps', 100)
        self.nflush = self.cfg.getint(cfgsect, 'nflush', 10)

        self._set_external('var_p', 'scalar fpdtype_t')
        self.tprev = -1.0
        self.nstep_counter = 0
        self.nflush_counter = 0
        self.init = False
        self.elemap_copy = elemap

        if self.bccomm.rank == 0:
            self.outf = init_csv(self.cfg, self.cfgsect, 't,mf,pbc')

    def calculate_mass_flow(self, solns):
        ndims, nvars = self.ndims, self.nvars
        fm = np.zeros((ndims))
        # Get the sizes for the area calculation
        for etype, fidx in self._m0:
            # Get the interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)

            # Get the quadrature weights and normal vectors
            qwts = self._qwts[etype, fidx]
            norms = self._norms[etype, fidx]

            # Do the quadrature for each dimension
            # RhoU = ufpts[1], RhoV = ufpts[2], RhoW = ufpts[2]
            for i in range(0, ndims):
                rhoVel = ufpts[1 + i]
                fm[i] += np.einsum('i...,ij,ji', qwts, rhoVel, norms[:,:,i])
        self.bccomm.Allreduce(mpi.IN_PLACE, fm, op=mpi.SUM)
        return sum(fm)
    
    def update_mf(self, solns):
        mf = self.calculate_mass_flow(solns)
        self.mf_avg = self.mf_alpha * mf + (1.0 - self.mf_alpha) * self.mf_avg

    def update_p(self, dt):
        self.p += dt * self.eta * (1.0 - self.target_mfr / self.mf_avg)
    
    def bind_p(self, kerns):
        for k in kerns:
            kerns[k].bind(var_p=self.p)

    def prepare(self, system, ubank, t, kerns):
        # Check if first prepare call
        if not self.init:
            self._surf_init(system, self.elemap_copy, self.bcname)
            del self.elemap_copy
            self.init = True

        # Check if past tstart
        if t < self.tstart:
            self.bind_p(kerns)
            return

        if self.nstep_counter % self.nsteps == 0:
            solns = dict(zip(system.ele_types, system.ele_scal_upts(ubank)))
            # First update to begin history
            if self.tprev < 0.0:
                self.tprev = t
                self.mf_avg = self.calculate_mass_flow(solns)
                self.bind_p(kerns)
                return

            self.update_mf(solns)
            self.update_p(t - self.tprev)
            self.tprev = t

            # Output mass flow and pressure at outflow
            if self.bccomm.rank == 0:
                print(f'{t},{self.mf_avg},{self.p}', 
                      file=self.outf)
            self.nflush_counter = self.nflush_counter + 1
        self.bind_p(kerns)

        # Flush to file
        if self.nflush_counter % self.nflush == 0:
            if self.bccomm.rank == 0:
                self.outf.flush()
        self.nstep_counter = self.nstep_counter + 1


class EulerCharRiemInvMassFlowBCInters(MassFlowBCMixin, EulerBaseBCInters):
    type = 'char-riem-inv-mass-flow'
