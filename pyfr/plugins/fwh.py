from collections import namedtuple

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import npeval
from pyfr.plugins.base import BaseSolnPlugin, SurfaceRegionMixin, init_csv
from pyfr.quadrules.surface import SurfaceIntegrator
from pyfr.util import first


FWHSurfParams = namedtuple(
    'FWHSurfParams',
    ('eidxs', 'm0', 'nda', 'r_tilde_vec',  'r_star_inv', 'r_star_tilde_vec')
)


class FWHIntegrator(SurfaceIntegrator):
    def __init__(self, cfg, cfgsect, ndims, obsv_pts, qinf, elemap, surf_list):
        super().__init__(cfg, cfgsect, elemap, surf_list)

        self.ndims, self.obsv_pts, self.qinf = ndims, obsv_pts, qinf

        Minf = np.linalg.norm(self.qinf['M'])
        if Minf >= 1:
            raise ValueError('FWH farfield Mach number greater than 1')

        self.surf = {}
        for etype, fidx in self.eidxs:
            for i, eidx in enumerate(self.eidxs[etype, fidx]):
                m0 = self.m0[etype, fidx]
                qwts = self.qwts[etype, fidx]
                pnorm = self.norms[etype, fidx][i]
                ploc = self.locs[etype, fidx][i]

                nds = qwts[:, None]*pnorm.transpose(2, 0, 1)
                nds = nds.reshape(self.ndims, -1)
                dist = self._distances(ploc, Minf)

                self.surf[etype, fidx] = FWHSurfParams(eidx, m0, nds, *dist)
    
    def _distances(self, spts, Minf):
        surf_pts = spts.transpose(0, 2, 1).reshape(-1, self.ndims)

        gamma_inv = (1 - Minf**2)**0.5
        gamma = 1 / gamma_inv

        r_o = self.obsv_pts[None] - surf_pts[:, None]
        d = np.linalg.norm(r_o, axis=-1)
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


class FWHPlugin(SurfaceRegionMixin, BaseSolnPlugin):
    name = 'fwh'
    systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None, *args, **kwargs):
        super().__init__(intg, cfgsect, suffix)
        comm, rank, root = get_comm_rank_root()

        self.elementscls = intg.system.elementscls

        self.tstart = self.cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.t_last = -np.inf
        self.dt = self.cfg.getfloat(cfgsect, 'dt')
        obsv_pts = np.array(self.cfg.getliteral(self.cfgsect, 'observer-pts'))
        self.nobvs = len(obsv_pts)

        # Initialise data file
        if rank == root:
            header = ','.join(['t', 'x', 'y', 'z'][:self.ndims + 1] + ['mag'])
            self.outf = init_csv(self.cfg, cfgsect, header)

        # Far field conditions
        self.incomp = intg.system.name in {'ac-euler', 'ac-navier-stokes'}
        privars = first(intg.system.ele_map.values()).privars
        self._vidx = [x in 'uvw' for x in privars]
        self._pidx = privars.index('p')
        self.consts = self.cfg.items_as('constants', float)

        qinf = {k: npeval(self.cfg.getexpr(cfgsect, k), self.consts) 
                for k in privars}
        self.uinf = np.array([[qinf[k]] for k in 'uvw'[:self.ndims]])

        if self.incomp:
            qinf['rho'] = self.cfg.getfloat(cfgsect, 'rho')
            qinf['c'] = self.cfg.getfloat(cfgsect, 'c')
        else:
            gamma = self.consts['gamma']
            qinf['c'] = (gamma * qinf['p'] / qinf['rho'])**0.5
            self._ridx = privars.index('rho')

        qinf['M'] = np.array([qinf[k] / qinf['c'] for k in 'uvw'[:self.ndims]])

        # Initialise surface data
        ele_map = intg.system.ele_map
        self.emap = {k: i for i, k in enumerate(ele_map)}
        ele_surf, _ = self._surf_region(intg)

        surfs = [(etype, fidx, eidxs) for _, etype, fidx, eidxs in ele_surf]
        self.fwh_int = FWHIntegrator(self.cfg, cfgsect, self.ndims, obsv_pts,
                                     qinf, ele_map, surfs)

        # Get boundary type info
        sname = self.cfg.get(cfgsect, 'surface')
        if '(' not in sname:
            self.bctype = self.cfg.get(f'soln-bcs-{sname}', 'type')
        else:
            self.bctype = None

    def _enforce_noslip_bc(self, pris):
        vmag = np.sum(pris[self._vidx]**2, axis=0)
        pris[self._vidx] = 0

        if not self.incomp:
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
                self._apply_wall_bc(pris)

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
        if intg.tcurr < self.tstart:
            return

        comm, rank, root = get_comm_rank_root()

        if intg.tcurr - self.dt >= self.t_last - self.tol:
            self.t_last = intg.tcurr

            o_vals = self._fwh_solve(intg)

            if rank != root:
                comm.Reduce(o_vals, None, op=mpi.SUM, root=root)
            else:
                comm.Reduce(mpi.IN_PLACE, o_vals, op=mpi.SUM, root=root)

                for x, p in zip(self.fwh_int.obsv_pts, o_vals):
                    print(intg.tcurr, *x, p, sep=',', file=self.outf)

                # Flush to disk
                self.outf.flush()
