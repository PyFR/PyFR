from pyfr.mpiutil import mpi, scal_coll
from pyfr.plugins.solver.nirf import (nirf_bc_params, nirf_origin_tplargs,
                                       _to_tplkey, _to_extern)
from pyfr.quadrules.surface import SurfaceIntegrator
from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)
from pyfr.util import CSVStream, first

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

    @staticmethod
    def common_pri_state(pl, nl, c):
        # Slip wall: remove the normal velocity component
        vel = np.asarray(pl[1:-1])
        un = np.linalg.vecdot(vel, nl, axis=0)

        return [pl[0], *(vel - un*nl), pl[-1]]


class ControlledBCMixin:
    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs
        )

        self.con_to_pri = first(elemap.values()).con_to_pri

        self.tstart = cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.nsteps = cfg.getint(cfgsect, 'nsteps', 100)

        opts = self._eval_opts(self._target_opts)
        self.target, self.alpha, self.eta = opts

        self.set_external('ic', 'scalar fpdtype_t')
        self.set_external('im', 'scalar fpdtype_t')

        self.surf_int = SurfaceIntegrator(cfg, cfgsect, elemap, lhs)

        self._init_extra(cfg, cfgsect)

        if cfg.hasopt(cfgsect, 'file') and bccomm.rank == 0:
            fname = cfg.get(cfgsect, 'file')
            nflush = cfg.getint(cfgsect, 'flushsteps', 10)
            self.csv = CSVStream(fname, header=self._csv_header, nflush=nflush)
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

            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, self.nvars, -1).swapaxes(0, 1)

            qwts = self.surf_int.qwts[etype, fidx]
            norms = self.surf_int.norms[etype, fidx]

            yield ufpts, qwts, norms

    def setup(self, sdata, prevcfg):
        sect_eq = (prevcfg is not None and
                   self.cfg.sect_eq(prevcfg, self.cfgsect))

        if sdata is not None and sdata[4] != 0 and sect_eq:
            self.interp_c = sdata[0]
            self.interp_m = sdata[1]
            self.meas_avg = sdata[2]
            self.tprev = sdata[3]
            self.nstep_counter = sdata[4]
        else:
            self.interp_c = self._default_interp_c()
            self.interp_m = 0.0
            self.meas_avg = 0.0
            self.tprev = None
            self.nstep_counter = 0

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
                p0 = self.interp_m*t + self.interp_c

                # Compute corrected Riemann pressure
                p1 = self._correction(p0, dt)

                # Update interpolation coefficients
                self.interp_m = (p1 - p0) / dt
                self.interp_c = p0 - self.interp_m*t

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

    def _init_extra(self, cfg, cfgsect):
        self._qwts_norms = []

        for etype, fidx in self.surf_int.m0:
            qwts = self.surf_int.qwts[etype, fidx]
            norms = self.surf_int.norms[etype, fidx]
            self._qwts_norms.append(norms*qwts[:, None])

    def _default_interp_c(self):
        return self._eval_opts(['p'])[0]

    def _measure(self, solns):
        mf = 0.0

        for (u, *_), qn in zip(self._interp_face(solns), self._qwts_norms):
            mf += np.einsum('hij,hij', u[1:-1], qn)

        return scal_coll(self.bccomm.Allreduce, mf, op=mpi.SUM)

    def _correction(self, p0, dt):
        return p0 + dt*self.eta*(1 - self.target / self.meas_avg)


class EulerCharRiemInvMassFlowBCInters(MassFlowBCMixin, EulerBaseBCInters):
    type = 'char-riem-inv-mass-flow'


class PressureBCMixin(ControlledBCMixin):
    _target_opts = ['pressure', 'alpha', 'eta']
    _csv_header = 't,p,pbc'

    def _init_extra(self, cfg, cfgsect):
        area = 0.0
        self._qwts_nmag = []

        all_qwts = self.surf_int.qwts.values()
        all_norms = self.surf_int.norms.values()
        for qwts, norms in zip(all_qwts, all_norms):
            nmag = np.sqrt(np.einsum('hij,hij->ij', norms, norms))
            qwts_nmag = nmag*qwts[:, None]
            self._qwts_nmag.append(qwts_nmag)
            area += qwts_nmag.sum()

        self.area = scal_coll(self.bccomm.Allreduce, area, op=mpi.SUM)

    def _default_interp_c(self):
        return self.target

    def _measure(self, solns):
        p_num = 0.0

        for (u, *_), qnmag in zip(self._interp_face(solns), self._qwts_nmag):
            p = self.con_to_pri(u, self.cfg)[-1]
            p_num += np.einsum('ij,ij', p, qnmag)

        p_num = scal_coll(self.bccomm.Allreduce, p_num, op=mpi.SUM)

        return p_num / self.area

    def _correction(self, p0, dt):
        return p0 + dt*self.eta*(self.target - self.meas_avg)


class EulerCharRiemInvPressureBCInters(PressureBCMixin, EulerBaseBCInters):
    type = 'char-riem-inv-pressure'


class NIRFBCMixin:
    """Mixin to transform BC velocities for NIRF rotating reference frame.

    Transforms inertial-frame velocities to body-frame:
        u_body = R(-θ) · (u_inertial - V₀) - Ω×r_body
    """
    nirf_section = 'solver-plugin-nirf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = self.nirf_section
        motion = self.cfg.get(sect, 'motion', 'prescribed')
        subs = self.cfg.items('constants')
        subs |= dict(abs='fabs', pi='3.141592653589793')

        params = nirf_bc_params(self.ndims)

        self._tplargs |= {
            _to_tplkey(p): _to_extern(p) if motion == 'free'
            else self.cfg.getexpr(sect, p, '0.0', subs=subs)
            for p in params
        }
        self._tplargs |= nirf_origin_tplargs(self.cfg, sect, self.ndims)

        if motion == 'free':
            for p in params:
                self.set_external(_to_extern(p), 'scalar fpdtype_t')

    def _exp_opts(self, opts, lhs, default={}):
        exprs = super()._exp_opts(opts, lhs, default)

        if not self.cfg.hasopt(self.nirf_section, 'motion'):
            return exprs

        sect = self.nirf_section
        motion = self.cfg.get(sect, 'motion')
        subs = self.cfg.items('constants')
        subs |= dict(abs='fabs', pi='3.141592653589793')

        def get_param(name):
            if motion == 'free':
                return _to_extern(name)
            return self.cfg.getexpr(sect, name, '0.0', subs=subs)

        fomega = [get_param(f'frame-omega-{c}') for c in 'xyz']
        floc = [get_param(f'frame-loc-{c}') for c in 'xyz'[:self.ndims]]
        fvelo = [get_param(f'frame-velo-{c}') for c in 'xyz'[:self.ndims]]
        fx0 = self.cfg.getliteral(sect, 'center-of-rot', (0.0,) * self.ndims)

        # Replace x,y,z with inertial locaiton
        for k in exprs:
            for i in range(self.ndims):
                exprs[k] = exprs[k].replace(
                    f'ploc[{i}]', f'(ploc[{i}] + ({floc[i]}))')

        # Step 1: inertial freestream minus frame translation
        u_inertial = [exprs.get(c) for c in 'uvw'[:self.ndims]]
        u_rel = [f'({ui}) - ({vi})' for ui, vi in zip(u_inertial, fvelo)]

        # Step 2: rotate to body frame using R(-θ) matrix
        u_body = [
            ' + '.join(f'({u_rel[j]})*nirf_R[{i}][{j}]'
                       for j in range(self.ndims))
            for i in range(self.ndims)
        ]

        # Step 3: subtract Ω×r (body coords)
        # Position relative to frame origin
        r = [f'(ploc[{i}] - ({fx0[i]}))' for i in range(self.ndims)]

        if self.ndims == 2:
            # 2D: -Ω×r = (ωz·ry, -ωz·rx)
            exprs['u'] = f'(({u_body[0]}) + ({fomega[2]})*{r[1]})'
            exprs['v'] = f'(({u_body[1]}) - ({fomega[2]})*{r[0]})'
        else:
            # 3D: -Ω×r = (-ωy·rz + ωz·ry, -ωz·rx + ωx·rz, -ωx·ry + ωy·rx)
            exprs['u'] = f'(({u_body[0]}) - ({fomega[1]})*{r[2]} + ({fomega[2]})*{r[1]})'
            exprs['v'] = f'(({u_body[1]}) - ({fomega[2]})*{r[0]} + ({fomega[0]})*{r[2]})'
            exprs['w'] = f'(({u_body[2]}) - ({fomega[0]})*{r[1]} + ({fomega[1]})*{r[0]})'

        # Register ploc for Ω×r
        if 'ploc' not in self._external_args:
            spec = f'in fpdtype_t[{self.ndims}]'
            value = self._const_mat(lhs, 'get_ploc_for_inters')
            self.set_external('ploc', spec, value=value)

        return exprs


class EulerCharRiemInvNIRFBCInters(NIRFBCMixin, EulerBaseBCInters):
    type = 'char-riem-inv-nirf'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class EulerSlpAdiaWallNIRFBCInters(NIRFBCMixin, EulerBaseBCInters):
    type = 'slp-adia-wall-nirf'
