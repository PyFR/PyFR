from collections import namedtuple

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import npeval
from pyfr.plugins.base import BaseSolnPlugin, SurfaceMixin, init_csv


FWHSurfParams = namedtuple('FWHSurfParams', ('eidxs', 'm0', 'norm', 'area',
                                             'r_tilde_vec', 'r_star_inv',
                                             'r_star_tilde_vec'))


class FWHPlugin(SurfaceMixin, BaseSolnPlugin):
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
        self.obsv_pts = np.array(self.cfg.getliteral(self.cfgsect,
                                                     'observer-pts'))
        self.nobvs = len(self.obsv_pts)

        # Initialise data file
        ndims = self.ndims
        if rank == root:
            header = ','.join(['t', 'x', 'y', 'z'][:self.ndims + 1] + ['mag'])
            self.outf = init_csv(self.cfg, cfgsect, header)

        # Far field conditions
        self.incomp = intg.system.name in {'ac-euler', 'ac-navier-stokes'}
        privarmap = intg.system.elementscls.privarmap[ndims]
        self._vidx = [x in 'uvw' for x in privarmap]
        self._pidx = privarmap.index('p')
        self.consts = self.cfg.items_as('constants', float)

        self.qinf = {k: npeval(self.cfg.getexpr(cfgsect, k), self.consts)
                     for k in privarmap}
        self.uinf = np.array([[self.qinf[k]] for k in 'uvw'[:ndims]])

        if self.incomp:
            self.qinf['rho'] = self.cfg.getfloat(cfgsect, 'rho')
            self.qinf['c'] = self.cfg.getfloat(cfgsect, 'c')
        else:
            gamma = self.consts['gamma']
            self.qinf['c'] = (gamma * self.qinf['p'] / self.qinf['rho'])**0.5

        self.qinf['M'] = np.array([self.qinf[k]/self.qinf['c'] for k in
                                   'uvw'[:ndims]])
        self.Minf = np.linalg.norm(self.qinf['M'])
        if self.Minf >= 1:
            raise ValueError('FWH farfield Mach number greater than 1')

        # Initialise surface data
        ele_map = intg.system.ele_map
        self.emap = {k: i for i, k in enumerate(ele_map)}
        self.ele_surface, _ = self._surf_region(intg)
        self._init_surf(ele_map)

    def _init_surf(self, ele_map):
        self.surf = {}
        for doff, etype, fidx, eidxs in self.ele_surface:
            eles = ele_map[etype]

            # Get face operators
            itype, proj, norm = eles.basis.faces[fidx]

            # Create a quadratue on suface
            qpts, qwts = self._surf_quad(itype, proj)
            norm = np.array([norm for x in qpts])

            # Get phyical location, transformations, and normals
            ploc = eles.ploc_at_np(qpts)[..., eidxs]
            rcpdjac = eles.rcpdjac_at_np(qpts)[..., eidxs]
            pnorm = eles.pnorm_at(qpts, norm)[:, eidxs]

            # Get ops and components of the surface
            m0 = eles.basis.ubasis.nodal_basis_at(qpts)
            norn_mag = np.linalg.norm(pnorm, axis=-1, keepdims=True)
            n = (pnorm / norn_mag).transpose(2, 0, 1).reshape(self.ndims, -1)
            area = (qwts[:, None] / rcpdjac).reshape((-1))
            dist = self._distances(ploc)

            self.surf[etype, fidx] = FWHSurfParams(eidxs, m0, n, area, *dist)

    def _distances(self, spts):
        surf_pts = spts.transpose(0, 2, 1).reshape(-1, self.ndims)

        gamma_inv = (1 - self.Minf**2)**0.5
        gamma = 1 / gamma_inv

        r_o = self.obsv_pts[None] - surf_pts[:, None]
        d = np.linalg.norm(r_o, axis=-1)
        r_o_hat = r_o / d[..., None]

        m_r = r_o_hat @ self.qinf['M']

        r_star_vec = r_o*np.hypot(gamma_inv, m_r)[..., None]
        r_star_inv = 1 / np.linalg.norm(r_star_vec, axis=-1)

        r_grad_fac = (np.einsum('ij,k->ijk', m_r, self.qinf['M']) +
                      r_o_hat*gamma_inv**2)
        r_snorm = r_o*r_star_inv[..., None]

        r_star_tilde_vec = r_snorm*r_grad_fac
        r_tilde_vec = (r_snorm - self.qinf['M'])*gamma**2

        return r_tilde_vec, r_star_inv, r_star_tilde_vec

    def _fwh_solve(self, intg):
        o_vals = np.zeros(self.nobvs)
        ci = 1 / self.qinf['c']

        # Query dt_soln to prevent MPI deadlock
        dt_soln = intg.dt_soln

        # Accumulate FWH contribution from each surface part
        for (etype, fidx), param in self.surf.items():
            soln = intg.soln[self.emap[etype]][..., param.eidxs]
            soln_t = dt_soln[self.emap[etype]][..., param.eidxs]

            s = param.m0 @ soln.transpose(1, 0, 2)
            s_t = param.m0 @ soln_t.transpose(1, 0, 2)

            pris = self.elementscls.con_to_pri(s, self.cfg)
            pris = np.reshape(pris, (self.nvars, -1))

            p = pris[self._pidx] - self.qinf['p']
            u = pris[self._vidx] - self.uinf
            d_inf = self.qinf['rho']
            d_tot = d_inf + p*ci**2

            mom = d_tot*(u + self.uinf)
            drift = -d_inf*self.uinf

            # Time derivatives
            pris_t = self.elementscls.diff_con_to_pri(s, s_t, self.cfg)
            pris_t = np.reshape(pris_t, (self.nvars, -1))

            u_t = pris_t[self._vidx]
            p_t = pris_t[self._pidx]

            d_tot_t = p_t / self.qinf['c']**2
            mom_t = d_tot_t*(u + self.uinf) + d_tot*u_t
            mom_t_n = np.sum(param.norm*mom_t, axis=0, keepdims=True)

            # Monopole
            q = np.sum(param.norm*(mom + drift), axis=0, keepdims=True).T
            q_t = mom_t_n.T
            acc = (1 - self.Minf)*q_t*param.r_star_inv
            acc -= q*param.r_star_inv**2*(
                param.r_star_tilde_vec @ self.uinf.reshape(-1)
            )

            # Dipole
            mom_n = np.sum(param.norm*mom, axis=0, keepdims=True)
            f = mom_n*u + (p - self.qinf['p'])*param.norm
            f_t = mom_t_n*u + mom_n*u_t + p_t*param.norm

            acc += ci*param.r_star_inv*np.einsum(
                'ki,ijk->ij',
                f_t,
                param.r_tilde_vec
            )
            acc += param.r_star_inv**2*np.einsum(
                'ki,ijk->ij',
                f,
                param.r_star_tilde_vec
            )

            # Quadrature and accumulate
            o_vals += param.area @ acc

        return o_vals / (4*np.pi)

    def __call__(self, intg):
        if intg.tcurr < self.tstart:
            return

        comm, rank, root = get_comm_rank_root()

        if intg.tcurr - self.dt >= self.t_last - self.tol:
            self.t_last = intg.tcurr

            o_vals = self._fwh_solve(intg)
            comm.Barrier()

            if rank != root:
                comm.Reduce(o_vals, None, op=mpi.SUM, root=root)
            else:
                comm.Reduce(mpi.IN_PLACE, o_vals, op=mpi.SUM, root=root)

                for x, p in zip(self.obsv_pts, o_vals):
                    print(intg.tcurr, *x, p, sep=',', file=self.outf)

                # Flush to disk
                self.outf.flush()
