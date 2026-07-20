from collections import namedtuple

from pyfr.mpiutil import mpi, scal_coll
from pyfr.quadrules.surface import SurfaceIntegrator
from pyfr.readers.native import Connectivity
from pyfr.util import first
from pyfr.writers.csv import CSVStream

import numpy as np


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


_NSCBCFace = namedtuple('_NSCBCFace', ('lhs', 'tplargs', 'extern_args',
                                       'extern_vals', 'kdata'))


class NSCBCMixin:
    flip_norm = False
    _nscbc_kern = 'pyfr.solvers.euler.kernels.bccflux_nscbc'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # NSCBC replaces comm_flux with its own corrected flux, scheduled
        # after the other interface fluxes
        self.kernels.pop('comm_flux', None)
        self._be.pointwise.register(self._nscbc_kern)
        self.kernels['nscbc_flux'] = lambda: self.gen_nscbc_kerns()

        # One kernel is emitted per element-face pair on the boundary
        self.nscbc_faces = []
        for etype, fidx, eidxs, idx in lhs.foreach():
            lhs_efp = Connectivity(lhs.cidxs[idx], lhs.eidxs[idx], lhs.cidxmap)
            self.nscbc_faces.append(self._build_face(etype, fidx, lhs_efp))

    def _build_face(self, etype, fidx, lhs):
        ele = self.elemap[etype]
        basis = ele.basis
        ndims = self.ndims
        nupts, nfpts = basis.nupts, basis.nfpts
        nfacefpts = basis.nfacefpts[fidx]
        facefpts = basis.facefpts[fidx]
        allfpts = [i for ff in basis.facefpts for i in ff]
        intfpts = [i for i in allfpts if i not in facefpts]

        # The transformed flux picks up a viscous part for Navier-Stokes
        visc = hasattr(ele, '_grad_upts')

        # Unit reference normals at the face flux points
        magn = np.linalg.norm(basis.norm_fpts, axis=-1)
        norm_ref = basis.norm_fpts[facefpts] / magn[facefpts, None]
        if self.flip_norm:
            norm_ref = -norm_ref

        # Correction-function divergences used to invert the FR update
        gbasis_fpts = basis.gbasis_at(basis.fpts)
        GB = gbasis_fpts[np.ix_(facefpts, facefpts)]
        GI = gbasis_fpts[np.ix_(facefpts, intfpts)]
        GB_inv = np.linalg.inv(GB)

        # Element geometry restricted to this pair's elements
        eidxs = lhs.eidxs
        smats = ele.smat_at_np('upts')[..., eidxs]
        smats = smats.transpose(1, 0, 2, 3).reshape(nupts, ndims*ndims, -1)
        jacs = 1.0 / ele.rcpdjac_at_np('fpts')[facefpts][:, eidxs]

        # The transformed-flux evaluation reads the copied solution at upts
        ele._soln_copy_required = True

        jac_fpts = np.rollaxis(basis.ubasis.jac_nodal_basis_at(basis.fpts), 2)
        m0 = basis.m0[facefpts]
        m2 = basis.m2.reshape(nfpts, ndims*nupts)
        m_div = jac_fpts[facefpts].reshape(nfacefpts, ndims*nupts)

        kdata = dict(
            u_upts=self._ewise_view(etype, eidxs, '_scal_upts_cpy',
                                    (nupts, self.nvars)),
            u_fpts=self._ewise_view(etype, eidxs, '_scal_fpts',
                                    (nfpts, self.nvars)),
            smats_upts=self._be.const_matrix(smats),
            jacs_ffpts=self._be.const_matrix(jacs),
            m0=self._be.const_matrix(m0),
            m2=self._be.const_matrix(m2),
            m_div=self._be.const_matrix(m_div)
        )
        if visc:
            kdata['gradu_upts'] = self._ewise_view(etype, eidxs, '_grad_upts',
                                                   (ndims*nupts, self.nvars))

        tplargs = self._tplargs | dict(
            nupts=nupts, nfpts=nfpts, nfacefpts=nfacefpts,
            facefpts=facefpts, intfpts=intfpts, norm_ref=norm_ref,
            GB_inv=GB_inv, GB_inv_GI=GB_inv @ GI, waves=self.waves
        )

        return _NSCBCFace(
            lhs=lhs, tplargs=tplargs,
            extern_args=self._external_args.copy(),
            extern_vals=self._external_vals.copy(),
            kdata=kdata
        )

    def _ewise_view(self, etype, eidxs, buf, vshape):
        mat = getattr(self.elemap[etype], buf)
        n = len(eidxs)
        return self._be.view(np.full(n, mat.mid), np.zeros(n, dtype=int),
                             eidxs, vshape=vshape)

    def gen_nscbc_kerns(self):
        kerns = [
            self._be.kernel('bccflux_nscbc', tplargs=f.tplargs,
                            dims=[len(f.lhs)], extrns=f.extern_args,
                            **f.kdata, **f.extern_vals)
            for f in self.nscbc_faces
        ]

        return self._be.unordered_meta_kernel(kerns)
