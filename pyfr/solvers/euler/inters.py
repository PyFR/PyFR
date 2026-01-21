from pyfr.mpiutil import mpi, scal_coll
from pyfr.quadrules.surface import SurfaceIntegrator
from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)
from pyfr.writers.csv import CSVStream

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

        self.tstart = cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.nsteps = cfg.getint(cfgsect, 'nsteps', 100)
        opts = self._eval_opts(['mass-flow-rate', 'alpha', 'eta'])
        self.target_mfr, self.alpha, self.eta = opts

        self._set_external('ic', 'scalar fpdtype_t')
        self._set_external('im', 'scalar fpdtype_t')

        surf_list = [(etype, fidx, eidx) for etype, eidx, fidx in lhs]
        self.mf_int = SurfaceIntegrator(cfg, cfgsect, elemap, surf_list)

        if cfg.hasopt(cfgsect, 'file') and bccomm.rank == 0:
            fname = cfg.get(cfgsect, 'file')
            nflush = cfg.getint(cfgsect, 'flushsteps', 10)
            self.csv = CSVStream(fname, header='t,mf,pbc', nflush=nflush)
        else:
            self.csv = None

    def setup(self, sdata):
        if sdata is not None and sdata[4] != 0:
            (self.interp_c, self.interp_m, 
            self.mf_avg, self.tprev, self.nstep_counter) = sdata
        else:
            self.interp_c = self._eval_opts(['p'])[0]
            self.interp_m = 0.0
            self.mf_avg = 0.0
            self.tprev = None
            self.nstep_counter = 0

    def calculate_mass_flow(self, solns):
        mf = 0.0

        for etype, fidx in self.mf_int.m0:
            # Get the interpolation operator
            m0 = self.mf_int.m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements and variables from the solution
            uupts = solns[etype][:, 1:-1, self.mf_int.eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, self.ndims, -1)

            # Get the quadrature weights and normal vectors
            qwts = self.mf_int.qwts[etype, fidx]
            norms = self.mf_int.norms[etype, fidx]

            # Do the quadrature
            mf += np.einsum('i,ihj,jih', qwts, ufpts, norms)

        return scal_coll(self.bccomm.Allreduce, mf, op=mpi.SUM)

    @classmethod
    def preparefn(cls, bciface, mesh, elemap):
        if bciface:
            return bciface.prepare
        else:
            return None

    def prepare(self, system, ubank, t, kerns):
        update = self.nstep_counter % self.nsteps == 0
        if (update or not self.tprev) and t >= self.tstart:
            solns = dict(zip(system.ele_types, system.ele_scal_upts(ubank)))
            mf = self.calculate_mass_flow(solns)

            if not self.tprev:
                self.mf_avg = mf
                self.tprev = t
            else:
                self.mf_avg = self.alpha * mf + (1 - self.alpha) * self.mf_avg
                dt = (t - self.tprev)
                self.tprev = t

                # Current p
                p0 = self.interp_m * t + self.interp_c

                # Next target p
                p1 = p0 + dt * self.eta * (1 - self.target_mfr / self.mf_avg)

                # Update interpolation
                self.interp_m = (p1 - p0) / dt
                self.interp_c = p0 - self.interp_m * t

                # Output mass flow and pressure at BC
                if self.csv:
                    self.csv(t, self.mf_avg, p1)

        # Bind interpolation to kernels
        for k in kerns.values():
            k.bind(ic=self.interp_c, im=self.interp_m)

        self.nstep_counter += 1
    
    @classmethod
    def serialisefn(cls, bciface, prefix, srl):
        srl.register(prefix, bciface._sdata if bciface else None)
    
    def _sdata(self):
        return np.void((self.interp_c, self.interp_m, self.mf_avg,
                        self.tprev or 0, self.nstep_counter),
                        dtype='f8,f8,f8,f8,i8')


class EulerCharRiemInvMassFlowBCInters(MassFlowBCMixin, EulerBaseBCInters):
    type = 'char-riem-inv-mass-flow'
