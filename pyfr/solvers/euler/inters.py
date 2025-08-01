from pyfr.mpiutil import mpi
from pyfr.plugins.base import init_csv
from pyfr.quadrules.surface import SurfaceIntegrator
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


class MassFlowBCMixin:
    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs
        )

        self.target_mfr = cfg.getfloat(cfgsect, 'mass-flow-rate')
        self.tstart = cfg.getfloat(cfgsect, 'tstart')
        self.p = cfg.getfloat(cfgsect, 'p')
        self.bcname = cfgsect.removeprefix('soln-bcs-')
        self.alpha = cfg.getfloat(cfgsect, 'alpha')
        self.eta = cfg.getfloat(cfgsect, 'eta')
        self.nsteps = cfg.getint(cfgsect, 'nsteps', 100)
        self.nflush = cfg.getint(cfgsect, 'nflush', 10)

        self._set_external('var_p', 'scalar fpdtype_t')
        self.tprev = None
        self.nstep_counter = 0
        self.nflush_counter = 0
        self.mf_avg = 0.0
        
        surf_list = [(etype, fidx, eidxs) for etype, eidxs, fidx in lhs]
        self.mf_int = SurfaceIntegrator(cfg, cfgsect, elemap, surf_list)

        self.writecsv = cfg.hasopt(cfgsect, 'file')
        if self.writecsv and bccomm.rank == 0:
            self.outf = init_csv(cfg, cfgsect, 't,mf,pbc')

    def calculate_mass_flow(self, solns):
        ndims, nvars = self.ndims, self.nvars
        mf = 0.0

        # Get the sizes for the area calculation
        for etype, fidx in self.mf_int.m0:
            # Get the interpolation operator
            m0 = self.mf_int.m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self.mf_int.eidxs[etype, fidx]]
            # Remove unused variables
            uupts = uupts[:,1:-1]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars-2, -1)
            ufpts = ufpts.swapaxes(0, 1)

            # Get the quadrature weights and normal vectors
            qwts = self.mf_int.qwts[etype, fidx]
            norms = np.rollaxis(self.mf_int.norms[etype, fidx], 2)

            # Do the quadrature for each dimension
            for _ufpts, _norms in zip(ufpts, norms):
                mf += np.einsum('i...,ij,ji', qwts, _ufpts, _norms)

        self.bccomm.allreduce(mf, op=mpi.SUM)
        return float(mf)

    def prepare(self, system, ubank, t, kerns):
        if t >= self.tstart and not self.tprev:
            # First update to begin history
            solns = dict(zip(system.ele_types, system.ele_scal_upts(ubank)))
            self.mf_avg = self.calculate_mass_flow(solns)
            self.tprev = t
        elif t >= self.tstart and self.nstep_counter % self.nsteps == 0:
            solns = dict(zip(system.ele_types, system.ele_scal_upts(ubank)))
            mf = self.calculate_mass_flow(solns)
            self.mf_avg = self.alpha * mf + (1.0 - self.alpha) * self.mf_avg

            dt = (t - self.tprev)
            if dt > 0.0:
                self.p += dt * self.eta * (1.0 - self.target_mfr / self.mf_avg)
                self.tprev = t

            # Output mass flow and pressure at BC
            if self.writecsv and self.bccomm.rank == 0:
                print(f'{t},{self.mf_avg},{self.p}', file=self.outf)

            self.nflush_counter += 1
        
        # Bind p to kernels
        for k in kerns.values():
            k.bind(var_p=self.p)

        # Flush to file
        if (self.writecsv and self.nflush_counter % self.nflush == 0 
            and self.bccomm.rank == 0):
            self.outf.flush()
        
        self.nstep_counter += 1


class EulerCharRiemInvMassFlowBCInters(MassFlowBCMixin, EulerBaseBCInters):
    type = 'char-riem-inv-mass-flow'
