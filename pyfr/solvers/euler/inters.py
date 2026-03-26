from pyfr.mpiutil import mpi, scal_coll
from pyfr.quadrules.surface import SurfaceIntegrator
from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)
from pyfr.writers.csv import CSVStream

import numpy as np


class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        if self.cfg.get('solver', 'shock-capturing', 'none') == 'entropy-filter':
            self.p_min = self.cfg.getfloat('solver-entropy-filter', 'p-min',
                                           1e-6)
        else:
            self.p_min = self.cfg.getfloat('solver-interfaces', 'p-min',
                                           5*self._be.fpdtype_eps)

        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, c=self.c, p_min=self.p_min)


class EulerIntInters(TplargsMixin, BaseAdvectionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.euler.kernels.intcflux')

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs, nl=self._pnorm_lhs
        )


class EulerMPIInters(TplargsMixin, BaseAdvectionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.euler.kernels.mpicflux')

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs, nl=self._pnorm_lhs
        )


class EulerBaseBCInters(TplargsMixin, BaseAdvectionBCInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.euler.kernels.bccflux')

        self._tplargs |= dict(bctype=self.type, ninters=self.ninters)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self.scal_lhs, nl=self._pnorm_lhs,
            **self._external_vals
        )

    def comm_entropy_kernel(self, entmin_lhs):
        # Physics-specific callback for entropy filtering
        self._be.pointwise.register('pyfr.solvers.euler.kernels.bccent')

        return lambda: self._be.kernel(
            'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, entmin_lhs=entmin_lhs,
            nl=self._pnorm_lhs, ul=self.scal_lhs, **self._external_vals
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


class ControlledBCMixin:
    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs
        )

        self.tstart = cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.nsteps = cfg.getint(cfgsect, 'nsteps', 100)

        opts = self._eval_opts(self._target_opts)
        self.target, self.alpha, self.eta = opts

        self.set_external('ic', 'scalar fpdtype_t')
        self.set_external('im', 'scalar fpdtype_t')

        surf_list = [(etype, fidx, eidx) for etype, eidx, fidx in lhs]
        self.surf_int = SurfaceIntegrator(cfg, cfgsect, elemap, surf_list)

        self._init_extra(cfg, cfgsect)

        if cfg.hasopt(cfgsect, 'file') and bccomm.rank == 0:
            fname = cfg.get(cfgsect, 'file')
            nflush = cfg.getint(cfgsect, 'flushsteps', 10)
            self.csv = CSVStream(fname, header=self._csv_header,
                                 nflush=nflush)
        else:
            self.csv = None

    def _init_extra(self, cfg, cfgsect):
        pass

    def _interp_face(self, solns):
        for etype, fidx in self.surf_int.m0:
            m0 = self.surf_int.m0[etype, fidx]
            nfpts, nupts = m0.shape

            eidxs = self.surf_int.eidxs[etype, fidx]
            uupts = solns[etype][:, :, eidxs]

            nv = uupts.shape[1]
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nv, -1).swapaxes(0, 1)

            qwts = self.surf_int.qwts[etype, fidx]
            norms = self.surf_int.norms[etype, fidx]

            yield ufpts, qwts, norms

    def setup(self, sdata, prevcfg):
        sect_eq = (prevcfg is not None and
                   self.cfg.sect_eq(prevcfg, self.cfgsect))

        if sdata is not None and sdata[4] != 0 and sect_eq:
            self.interp_c, self.interp_m = sdata[:2]
            self.meas_avg, self.tprev = sdata[2:4]
            self.nstep_counter = sdata[4]
        else:
            self.interp_c = self._default_interp_c()
            self.interp_m = 0.0
            self.meas_avg = 0.0
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
            mf += np.einsum('i,ihj,hij', qwts, ufpts, norms)

        return scal_coll(self.bccomm.Allreduce, mf, op=mpi.SUM)

    @classmethod
    def preparefn(cls, bciface, mesh, elemap):
        if bciface:
            return bciface.prepare
        else:
            return None

    def prepare(self, system, ubank, t, kerns):
        if not hasattr(self, 'elementscls'):
            self.elementscls = system.elementscls

        update = self.nstep_counter % self.nsteps == 0
        if (update or not self.tprev) and t >= self.tstart:
            solns = dict(zip(system.ele_types, system.ele_scal_upts(ubank)))
            meas = self._measure(solns)

            if not self.tprev:
                self.meas_avg = meas
                self.tprev = t
            else:
                a = self.alpha
                self.meas_avg = a*meas + (1 - a)*self.meas_avg
                dt = t - self.tprev
                self.tprev = t

                # Current Riemann invariant pressure
                p0 = self.interp_m * t + self.interp_c

                # Compute corrected Riemann pressure
                p1 = self._correction(p0, dt)

                # Update interpolation coefficients
                self.interp_m = (p1 - p0) / dt
                self.interp_c = p0 - self.interp_m * t

                # Log to CSV
                if self.csv:
                    self.csv(t, self.meas_avg, p1)

        # Bind interpolation to kernels
        for k in kerns.values():
            k.bind(ic=self.interp_c, im=self.interp_m)

        self.nstep_counter += 1

    @classmethod
    def serialisefn(cls, bciface, prefix, srl):
        sfn = lambda: np.void(
            (bciface.interp_c, bciface.interp_m, bciface.meas_avg,
             bciface.tprev or 0, bciface.nstep_counter),
            dtype='f8,f8,f8,f8,i8'
        )
        srl.register(prefix, sfn if bciface else None)


class MassFlowBCMixin(ControlledBCMixin):
    _target_opts = ['mass-flow-rate', 'alpha', 'eta']
    _csv_header = 't,mf,pbc'

    def _default_interp_c(self):
        return self._eval_opts(['p'])[0]

    def _measure(self, solns):
        mf = 0.0

        for ufpts, qwts, norms in self._interp_face(solns):
            mf += np.einsum('i,hij,jih', qwts, ufpts[1:-1], norms)

        return scal_coll(self.bccomm.Allreduce, mf, op=mpi.SUM)

    def _correction(self, p0, dt):
        return p0 + dt * self.eta * (1 - self.target / self.meas_avg)


class EulerCharRiemInvMassFlowBCInters(MassFlowBCMixin, EulerBaseBCInters):
    type = 'char-riem-inv-mass-flow'


class PressureBCMixin(ControlledBCMixin):
    _target_opts = ['target-pressure', 'alpha', 'eta']
    _csv_header = 't,p_avg,p_riem'

    def _init_extra(self, cfg, cfgsect):
        area = 0.0
        for etype, fidx in self.surf_int.m0:
            qwts = self.surf_int.qwts[etype, fidx]
            norms = self.surf_int.norms[etype, fidx]
            nmag = np.sqrt(np.einsum('jih,jih->ji', norms, norms))
            area += np.einsum('i,ji->', qwts, nmag)
        self.area = scal_coll(self.bccomm.Allreduce, area, op=mpi.SUM)

    def _default_interp_c(self):
        return self.target

    def _measure(self, solns):
        p_num = 0.0

        for ufpts, qwts, norms in self._interp_face(solns):
            p = self.elementscls.con_to_pri(ufpts, self.cfg)[-1]
            nmag = np.sqrt(np.einsum('jih,jih->ji', norms, norms))
            p_num += np.einsum('i,ij,ji', qwts, p, nmag)

        p_num = scal_coll(self.bccomm.Allreduce, p_num, op=mpi.SUM)

        return p_num / self.area

    def _correction(self, p0, dt):
        return p0 + dt * self.eta * (self.target - self.meas_avg)


class EulerCharRiemInvPressureBCInters(PressureBCMixin, EulerBaseBCInters):
    type = 'char-riem-inv-pressure'
