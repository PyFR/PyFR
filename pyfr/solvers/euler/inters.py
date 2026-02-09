from pyfr.mpiutil import mpi, scal_coll
from pyfr.plugins.nirf import (nirf_bc_params, nirf_origin_tplargs,
                                _to_tplkey, _to_extern)
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

        self.set_external('ic', 'scalar fpdtype_t')
        self.set_external('im', 'scalar fpdtype_t')

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
        sfn = lambda: np.void((bciface.interp_c, bciface.interp_m,
                               bciface.mf_avg, bciface.tprev or 0,
                               bciface.nstep_counter),
                              dtype='f8,f8,f8,f8,i8')
        srl.register(prefix, sfn if bciface else None)


class EulerCharRiemInvMassFlowBCInters(MassFlowBCMixin, EulerBaseBCInters):
    type = 'char-riem-inv-mass-flow'


class NIRFBCMixin:
    """Mixin to transform BC velocities for NIRF rotating reference frame.

    Transforms inertial-frame velocities to body-frame:
        u_body = R(-θ) · (u_inertial - V₀) - Ω×r_body

    For prescribed mode (no rotation, θ=0), R=I and this simplifies to:
        u_body = u_inertial - V₀ - Ω×r
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

            # Rotation matrix externs: R(-θ) for lab-to-body transform
            for i in range(3):
                for j in range(3):
                    self.set_external(f'nirf_R{i}{j}', 'scalar fpdtype_t')

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

        comps = 'xyz'[:self.ndims]
        fomega = [get_param(f'frame-omega-{c}') for c in 'xyz']
        fx0 = [self.cfg.getfloat(sect, f'frame-origin-{c}', 0.0)
               for c in comps]
        fvelo = [get_param(f'frame-velo-{c}') for c in comps]

        # Step 1: inertial freestream minus frame translation
        u_inertial = [exprs.get(c) for c in 'uvw'[:self.ndims]]
        u_rel = [f'({ui}) - ({vi})' for ui, vi in zip(u_inertial, fvelo)]

        # Step 2: rotate to body frame using R(-θ) matrix
        if motion == 'free':
            u_body = [
                ' + '.join(f'({u_rel[j]})*nirf_R{i}{j}'
                           for j in range(self.ndims))
                for i in range(self.ndims)
            ]
        else:
            u_body = u_rel

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
            value = self._const_mat(lhs, 'get_ploc_for_inter')
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
