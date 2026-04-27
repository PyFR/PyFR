from collections import defaultdict

import numpy as np

from pyfr.readers.native import Connectivity
from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)
from pyfr.solvers.euler.inters import MassFlowBCMixin, PressureBCMixin


class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'entropy-filter':
            self.p_min = self.cfg.getfloat('solver-entropy-filter', 'p-min',
                                           1e-6)
        else:
            self.p_min = self.cfg.getfloat('solver-interfaces', 'p-min',
                                           5*self._be.fpdtype_eps)

        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, visc_corr=visc_corr,
                             shock_capturing=shock_capturing, c=self.c,
                             p_min=self.p_min)

class NavierStokesIntInters(TplargsMixin,
                            BaseAdvectionDiffusionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self.scal_lhs, urin=self.scal_rhs,
            ulout=self._comm_lhs, urout=self._comm_rhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs
        )


class NavierStokesMPIInters(TplargsMixin,
                            BaseAdvectionDiffusionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'mpiconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self.scal_lhs, urin=self.scal_rhs, ulout=self._comm_lhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs
        )


class NavierStokesBaseBCInters(TplargsMixin, BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional BC specific template arguments
        self._tplargs['bctype'] = self.type
        self._tplargs['bccfluxstate'] = self.cflux_state

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'bcconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ulin=self.scal_lhs,
            ulout=self._comm_lhs, nlin=self._pnorm_lhs,
            **self._external_vals
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self.scal_lhs,
            gradul=self._vect_lhs, nl=self._pnorm_lhs,
            artvisc=self.artvisc, **self._external_vals
        )

    def comm_entropy_kernel(self, entmin_lhs):
        # Physics-specific callback for entropy filtering
        self._be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.bccent'
        )

        return lambda: self._be.kernel(
            'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, entmin_lhs=entmin_lhs,
            nl=self._pnorm_lhs, ul=self.scal_lhs, **self._external_vals
        )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c['cpTw'], = self._eval_opts(['cpTw'])
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost-imperm'


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        gamma = self.cfg.getfloat('constants', 'gamma')

        # Pass boundary constants to the backend
        self.c['cpTt'], = self._eval_opts(['cpTt'])
        self.c['pt'], = self._eval_opts(['pt'])
        self.c['Rdcp'] = (gamma - 1.0)/gamma

        # Calculate u, v velocity components from the inflow angle
        theta = self._eval_opts(['theta'])[0]*np.pi/180.0
        velcomps = np.array([np.cos(theta), np.sin(theta), 1.0])

        # Adjust u, v and calculate w velocity components for 3-D
        if self.ndims == 3:
            phi = self._eval_opts(['phi'])[0]*np.pi/180.0
            velcomps[:2] *= np.sin(phi)
            velcomps[2] *= np.cos(phi)

        self.c['vc'] = velcomps[:self.ndims]


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(['p'], lhs)


class NavierStokesCharRiemInvMassFlowBCInters(MassFlowBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-mass-flow'
    cflux_state = 'ghost'


class NavierStokesCharRiemInvPressureBCInters(PressureBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-pressure'
    cflux_state = 'ghost'


class NSCBCMixin:
    _nscbc_kern = 'pyfr.solvers.navstokes.kernels.bccflux_nscbc'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # NSCBC replaces the standard comm_flux with nscbc_flux
        self.kernels.pop('comm_flux', None)
        self._be.pointwise.register(self._nscbc_kern)

        self._tplargs_efp = defaultdict(dict)
        self._external_args_efp = defaultdict(dict)
        self._external_vals_efp = defaultdict(dict)
        self._scal_upts = defaultdict(dict)
        self._scal_fpts = defaultdict(dict)
        self._grad_upts = defaultdict(dict)
        self._smats_upts = defaultdict(dict)
        self._jacs_facefpts = defaultdict(dict)
        self._dim_lhs = defaultdict(dict)

        # Register NSCBC kernel under a different name so it can be scheduled after other BCs
        self.kernels['nscbc_flux'] = lambda: self.gen_nscbc_kerns()

        # Create sub-Connectivity for each element-face pair
        self.ef_pairs = []
        self._lhs_efp = defaultdict(dict)
        for etype, fidx, eidxs in lhs.items():
            self.ef_pairs.append((etype, fidx))
            cidx = next(c for c, (et, fi) in lhs.cidxmap.items()
                        if et == etype and fi == fidx)
            mask = lhs.cidxs == cidx
            self._lhs_efp[etype][fidx] = Connectivity(
                lhs.cidxs[mask], lhs.eidxs[mask], lhs.cidxmap
            )

        for etype, fidx in self.ef_pairs:
            lhs_efp = self._lhs_efp[etype][fidx]
            self._dim_lhs[etype][fidx] = len(lhs_efp)

            self._external_args_efp[etype][fidx] = self._external_args.copy()
            self._external_vals_efp[etype][fidx] = self._external_vals.copy()

            ele = self.elemap[etype]
            basis = ele.basis
            nupts = basis.nupts
            nfpts = basis.nfpts
            nfacefpts = basis.nfacefpts[fidx]
            ndims = self.ndims
            facefpts = basis.facefpts[fidx]
            fptidx = [i for j in basis.facefpts for i in j]
            intfpts = [i for i in fptidx if i not in facefpts]

            # Reference normal directions for computing physical normals
            magnl = np.linalg.norm(basis.norm_fpts, axis=-1)
            norms = basis.norm_fpts[facefpts] / magnl[facefpts, None]
            if '-in-' in self.type:
                norms = -norms

            # Correction function matrices for FR inversion
            # m11[i,j] is the divergence of correction function j at flux point i
            GB = basis.m11[np.ix_(facefpts, facefpts)]
            GI = basis.m11[np.ix_(facefpts, intfpts)]
            GB_inv = np.linalg.inv(GB)

            self._tplargs_efp[etype][fidx] = self._tplargs | dict(
                nupts=nupts, nfpts=nfpts, nfacefpts=nfacefpts,
                nintfacefpts=nfpts - nfacefpts, facefpts=facefpts,
                fptidx=fptidx, intfpts=intfpts,
                m0=basis.m0[facefpts],
                m2=basis.m2.reshape(nfpts, ndims, nupts),
                m12=basis.m12[facefpts], norm_ref=norms,
                GB_inv=GB_inv, GB_inv_GI=GB_inv @ GI,
                decomp_type=self.decomp_type,
            )

            self._scal_upts[etype][fidx] = self._scal_upts_view(
                lhs_efp, '_get_scal_upts_cpy_ewise')
            self._scal_fpts[etype][fidx] = self._scal_fpts_view(
                lhs_efp, '_get_scal_fpts_ewise')
            self._grad_upts[etype][fidx] = self._grad_upts_view(
                lhs_efp, '_get_grad_upts_ewise')

            # Collect smats at upts: (ndims, nupts, ndims, neles) -> (nupts, ndims*ndims, neles_efp)
            _, _, eidxs_efp = next(lhs_efp.items())
            smats = ele.smat_at_np('upts')[:, :, :, eidxs_efp]
            smats = smats.transpose(1, 0, 2, 3).reshape(nupts, ndims*ndims, -1)
            self._smats_upts[etype][fidx] = self._be.const_matrix(smats)

            # Collect jacs at face fpts: (nfacefpts, neles_efp)
            jacs = 1.0 / ele.rcpdjac_at_np('fpts')[facefpts][:, eidxs_efp]
            self._jacs_facefpts[etype][fidx] = self._be.const_matrix(jacs)

    def gen_nscbc_kerns(self):
        kerns = []
        for etype, fidx in self.ef_pairs:
            kerns.append(self._be.kernel(
                'bccflux_nscbc',
                tplargs=self._tplargs_efp[etype][fidx],
                dims=[self._dim_lhs[etype][fidx]],
                extrns=self._external_args_efp[etype][fidx],
                u_upts=self._scal_upts[etype][fidx],
                u_fpts=self._scal_fpts[etype][fidx],
                gradu_upts=self._grad_upts[etype][fidx],
                smats_upts=self._smats_upts[etype][fidx],
                jacs_ffpts=self._jacs_facefpts[etype][fidx],
                **self._external_vals_efp[etype][fidx]))

        return self._be.unordered_meta_kernel(kerns)


class NSCBCSubOutFpBCInters(NSCBCMixin, NavierStokesBaseBCInters):

    type = 'sub-out-nscbc-fp'
    decomp_type = 'normal'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        for etype, fidx in self.ef_pairs:
            lhs_efp = self._lhs_efp[etype][fidx]
            self.c |= self._exp_opts_ele(['p'], lhs_efp,
                                         self._external_args_efp[etype][fidx],
                                         self._external_vals_efp[etype][fidx])
        self.c['K_p'] = self.cfg.getfloat(cfgsect, 'K_p', default=0.25)

class NSCBCSubInFrvBCInters(NSCBCMixin, NavierStokesBaseBCInters):

    type = 'sub-in-nscbc-frv'
    decomp_type = 'cartesian'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)
        for etype, fidx in self.ef_pairs:
            lhs_efp = self._lhs_efp[etype][fidx]

            self.c |= self._exp_opts_ele(
                ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs_efp,
                self._external_args_efp[etype][fidx],
                self._external_vals_efp[etype][fidx],
            )
        for i in ['rho', 'u', 'v', 'w'][:self.ndims + 1]:
            self.c[f'K_{i}'] = self.cfg.getfloat(cfgsect, f'K_{i}', default=0.25)

class NSCBCSubInNRIBCInters(NSCBCMixin, NavierStokesBaseBCInters):

    type = 'sub-in-nscbc-nri'
    decomp_type = 'normal'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        force = ['u_a', 'du_a_dt', 'u_v', 'du_v_dt']
        for etype, fidx in self.ef_pairs:
            lhs_efp = self._lhs_efp[etype][fidx]

            self.c |= self._exp_opts_ele(
                ['rho', 'un'], lhs_efp,
                self._external_args_efp[etype][fidx],
                self._external_vals_efp[etype][fidx],
            )
            self.c |= self._exp_opts_ele(
                force, lhs_efp,
                self._external_args_efp[etype][fidx],
                self._external_vals_efp[etype][fidx],
                default={f: 0.0 for f in force},
            )
        for i in ['ac', 'ut']:
            self.c[f'K_{i}'] = self.cfg.getfloat(cfgsect, f'K_{i}', default=0.25)
