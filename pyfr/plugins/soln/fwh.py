from collections import namedtuple

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.exprs import npeval
from pyfr.plugins.mixins import SeriesWriterMixin, SurfaceRegionMixin
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.quadrules.surface import SurfaceIntegrator
from pyfr.util import expand_braces, first


FWHSurfParams = namedtuple(
    'FWHSurfParams',
    ('eidxs', 'm0', 'nda', 'r_tilde_vec',  'r_star_inv', 'r_star_tilde_vec')
)


class FWHIntegrator(SurfaceIntegrator):
    def __init__(self, cfg, cfgsect, ndims, obsv_pts, qinf, elemap, con):
        super().__init__(cfg, cfgsect, elemap, con)

        self.ndims, self.obsv_pts, self.qinf = ndims, obsv_pts, qinf

        Minf = np.linalg.norm(self.qinf['M'])
        if Minf >= 1:
            raise ValueError('FWH farfield Mach number greater than 1')

        self.surf = {}
        for etype, fidx in self.eidxs:
            eidxs = self.eidxs[etype, fidx]
            m0 = self.m0[etype, fidx]
            qwts = self.qwts[etype, fidx]
            norms = self.norms[etype, fidx]
            locs = self.locs[etype, fidx]

            nda = (qwts[None, :, None]*norms).reshape(ndims, -1)
            spts = locs.reshape(ndims, -1).T
            dist = self._distances(spts, Minf)

            self.surf[etype, fidx] = FWHSurfParams(eidxs, m0, nda, *dist)

    def _distances(self, surf_pts, Minf):
        gamma_inv = (1 - Minf**2)**0.5
        gamma = 1 / gamma_inv

        r_o = self.obsv_pts[None] - surf_pts[:, None]
        d = np.linalg.norm(r_o, axis=-1)

        if d.min() < 1e-8*d.max():
            raise ValueError('FWH observer point lies on the surface')

        r_o_hat = r_o / d[..., None]

        m_r = r_o_hat @ self.qinf['M']

        r_star_vec = r_o*np.hypot(gamma_inv, m_r)[..., None]
        r_star_inv = 1 / np.linalg.norm(r_star_vec, axis=-1)

        r_grad_fac = (np.einsum('ij,k->ijk', m_r, self.qinf['M'])
                      + r_o_hat*gamma_inv**2)
        r_snorm = r_star_inv*d

        r_star_tilde_vec = r_snorm[..., None]*r_grad_fac
        r_tilde_vec = (r_star_tilde_vec - self.qinf['M'])*gamma**2

        return r_tilde_vec, r_star_inv, r_star_tilde_vec


class FWHPlugin(SeriesWriterMixin, SurfaceRegionMixin, BaseSolnPlugin):
    name = 'fwh'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None, *args, **kwargs):
        super().__init__(intg, cfgsect, suffix)
        comm, rank, root = get_comm_rank_root()

        self.elementscls = intg.system.elementscls

        self.dt = self.cfg.getfloat(cfgsect, 'dt')
        obsv_pts = np.array(self.cfg.getliteral(self.cfgsect, 'observer-pts'))
        self.nobvs = len(obsv_pts)

        # Skip the restart time itself; it was sampled by the previous run
        self.t_last = intg.tcurr if intg.isrestart else -np.inf

        # Initialise data file
        if rank == root:
            self._init_series(intg, ['p'], obsv_pts)

        # Far field conditions
        privars = first(intg.system.ele_map.values()).privars
        self._vidx = [x in 'uvw' for x in privars]
        self._pidx = privars.index('p')
        self.consts = self.cfg.items_as('constants', float)

        qinf = {k: npeval(self.cfg.getexpr(cfgsect, k), self.consts)
                for k in privars}
        self.uinf = np.array([[qinf[k]] for k in 'uvw'[:self.ndims]])

        gamma = self.consts['gamma']
        qinf['c'] = (gamma * qinf['p'] / qinf['rho'])**0.5
        self._ridx = privars.index('rho')

        qinf['M'] = np.array([qinf[k] / qinf['c'] for k in 'uvw'[:self.ndims]])

        # Initialise surface data
        ele_map = intg.system.ele_map
        self.emap = {k: i for i, k in enumerate(ele_map)}
        con, _ = self._surf_region(intg)

        self.fwh_int = FWHIntegrator(self.cfg, cfgsect, self.ndims, obsv_pts,
                                     qinf, ele_map, con)

        # Solid boundary surfaces require the wall boundary treatment
        sname = self.cfg.get(cfgsect, 'surface')
        if sname.startswith('bc/'):
            bcsects = intg.system.mesh.bc_sections(self.cfg)
            bctypes = {self.cfg.get(bcsects[b], 'type')
                       for b in expand_braces(sname[3:])}

            # Check for malformed boundary groups
            if len(bctypes) > 1:
                raise ValueError(f'Surface {sname} spans mixed boundary '
                                 f'types: {sorted(bctypes)}')

            self.bctype = first(bctypes)
        else:
            self.bctype = None

    def _enforce_noslip_bc(self, pris):
        vmag = np.sum(pris[self._vidx]**2, axis=0)
        pris[self._vidx] = 0
        rho = pris[self._ridx]

        # Apply no-slip
        pris[self._pidx] += 0.5*(self.consts['gamma'] - 1)*rho*vmag

    def _fwh_solve(self, intg):
        o_vals = np.zeros(self.nobvs)
        ci = 1 / self.fwh_int.qinf['c']

        # Query dt_soln to prevent MPI deadlock
        dt_soln = intg.dt_soln

        # Accumulate FWH contribution from each surface part
        for (etype, fidx), param in self.fwh_int.surf.items():
            soln = intg.soln[self.emap[etype]][..., param.eidxs]
            soln_t = dt_soln[self.emap[etype]][..., param.eidxs]

            s = param.m0 @ soln.transpose(1, 0, 2)
            s_t = param.m0 @ soln_t.transpose(1, 0, 2)

            pris = self.elementscls.con_to_pri(s, self.cfg)
            pris = np.reshape(pris, (self.nvars, -1))

            if str(self.bctype).startswith('no-slp'):
                self._enforce_noslip_bc(pris)

            p = pris[self._pidx] - self.fwh_int.qinf['p']
            u = pris[self._vidx] - self.uinf
            d_inf = self.fwh_int.qinf['rho']
            d_tot = d_inf + p*ci**2

            mom = d_tot*(u + self.uinf)
            drift = -d_inf*self.uinf

            # Time derivatives
            pris_t = self.elementscls.diff_con_to_pri(s, s_t, self.cfg)
            pris_t = np.reshape(pris_t, (self.nvars, -1))

            u_t = pris_t[self._vidx]
            p_t = pris_t[self._pidx]

            d_tot_t = p_t / self.fwh_int.qinf['c']**2
            mom_t = d_tot_t*(u + self.uinf) + d_tot*u_t
            mom_t_n = np.sum(param.nda*mom_t, axis=0, keepdims=True)

            # Monopole
            q = np.sum(param.nda*(mom + drift), axis=0, keepdims=True).T
            q_t = mom_t_n.T
            acc = 1 - param.r_tilde_vec @ self.fwh_int.qinf['M']
            acc *= q_t*param.r_star_inv
            acc -= q*param.r_star_inv**2*(
                param.r_star_tilde_vec @ self.uinf.reshape(-1)
            )

            # Dipole
            mom_n = np.sum(param.nda*mom, axis=0, keepdims=True)
            f = mom_n*u + p*param.nda
            f_t = mom_t_n*u + mom_n*u_t + p_t*param.nda

            acc += ci*param.r_star_inv*np.einsum('ki,ijk->ij', f_t,
                                                 param.r_tilde_vec)
            acc += param.r_star_inv**2*np.einsum('ki,ijk->ij', f,
                                                 param.r_star_tilde_vec)

            # Accumulate
            o_vals += np.sum(acc, axis=0)

        return o_vals / (4*np.pi)

    def __call__(self, intg):
        comm, rank, root = get_comm_rank_root()

        if intg.tcurr - self.dt >= self.t_last - self.tol:
            self.t_last = intg.tcurr

            o_vals = self._fwh_solve(intg)

            if rank != root:
                comm.Reduce(o_vals, None, op=mpi.SUM, root=root)
            else:
                comm.Reduce(mpi.IN_PLACE, o_vals, op=mpi.SUM, root=root)

                self._write(intg.tcurr, o_vals)
