from pyfr.plugins.base import SurfaceMixin
from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)

class BCIntersSurfaceMixin(SurfaceMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Setup integrating over boundary
    def _init_surface_integration(self, system, elemap, bcname):
        # Underlying elements class
        self.elementscls = system.elementscls
        # Get the mesh and elements
        mesh = system.mesh
        # Interpolation matrices and quadrature weights
        self._m0 = m0 = {}
        self._qwts = qwts = defaultdict(list)
        # Element indices, associated face normals and relative flux
        # points position with respect to the moments origin
        eidxs = defaultdict(list)
        norms = defaultdict(list)
        rfpts = defaultdict(list)

        for etype, eidx, fidx in mesh.bcon[bcname]:
            eles = elemap[etype]
            itype, proj, norm = eles.basis.faces[fidx]

            ppts, pwts = self._surf_quad(itype, proj, flags='s')
            nppts = len(ppts)

            # Get phyical normals
            pnorm = eles.pnorm_at(ppts, [norm]*nppts)[:, eidx]

            eidxs[etype, fidx].append(eidx)
            norms[etype, fidx].append(pnorm)

            if (etype, fidx) not in m0:
                m0[etype, fidx] = eles.basis.ubasis.nodal_basis_at(ppts)
                qwts[etype, fidx] = pwts

        self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
        self._norms = {k: np.array(v) for k, v in norms.items()}
        self._rfpts = {k: np.array(v) for k, v in rfpts.items()}


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
