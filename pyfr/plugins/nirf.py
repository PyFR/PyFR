import math

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolverPlugin, init_csv
from pyfr.quadrules.surface import SurfaceIntegrator

# TODO: Add output options for prescribed


def nirf_src_params(ndims):
    comps = 'xyz'[:ndims]
    return ([f'frame-omega-{c}' for c in 'xyz'] +
            [f'frame-alpha-{c}' for c in 'xyz'] +
            [f'frame-accel-{c}' for c in comps])


def nirf_bc_params(ndims):
    comps = 'xyz'[:ndims]
    return ([f'frame-omega-{c}' for c in 'xyz'] +
            [f'frame-velo-{c}' for c in comps])


def nirf_origin_tplargs(cfg, cfgsect, ndims):
    return {f'frame_origin_{c}': cfg.getfloat(cfgsect, f'frame-origin-{c}', 0.0)
            for c in 'xyz'[:ndims]}


def _to_tplkey(p):
    return p.replace('-', '_') + '_expr'


def _to_extern(p):
    return p.replace('-', '_')


class NIRFForceIntegrator(SurfaceIntegrator):
    def __init__(self, cfg, cfgsect, system, bcname, morigin, viscous):
        surf_list = system.mesh.bcon.get(bcname, [])
        surf_list = [(etype, fidx, eidxs) for etype, eidxs, fidx in surf_list]

        super().__init__(cfg, cfgsect, system.ele_map, surf_list, flags='s')

        if surf_list and morigin is not None:
            self.rfpts = {k: loc - morigin for k, loc in self.locs.items()}

        # Set up m4 and rcpjact for local gradient computation
        if viscous:
            self.m4 = {}
            rcpjact = {}

            for etype in system.ele_map:
                eles = system.ele_map[etype]

                # Get m4 operator from element basis
                self.m4[etype] = eles.basis.m4

                # Get smat (scaling matrix) at solution points
                smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)

                # Get |J|^-1 at solution points
                rcpdjac = eles.rcpdjac_at_np('upts')

                # Compute J^-T at solution points
                rcpjact[etype] = smat * rcpdjac

            # Extract only boundary elements
            self.rcpjact = {k: rcpjact[k[0]][..., v]
                           for k, v in self.eidxs.items()}


class NIRFPlugin(BaseSolverPlugin):
    name = 'nirf'
    systems = ['euler', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self._intg = intg

        subs = self.cfg.items('constants')
        subs |= dict(abs='fabs', pi=math.pi)

        self._motion = self.cfg.get(cfgsect, 'motion')

        if self._motion == 'prescribed':
            self._init_prescribed(cfgsect, subs)
        elif self._motion == 'free':
            self._init_free(intg, cfgsect)
        else:
            raise ValueError(f"Invalid NIRF motion type: {self._motion}")

        macro = f'nirf_source_{self.ndims}d'
        for eles in intg.system.ele_map.values():
            eles.add_src_macro('pyfr.plugins.kernels.nirf', macro,
                               self._tplargs, ploc=True, soln=True)

    def _validate_prescribed_cfg(self, cfgsect):
        def is_const(key):
            try:
                float(self.cfg.get(cfgsect, key, '0.0'))
                return True
            except ValueError:
                return False

        pairs = [(f'frame-omega-{c}', f'frame-alpha-{c}') for c in 'xyz']
        pairs += [(f'frame-velo-{c}', f'frame-accel-{c}')
                  for c in 'xyz'[:self.ndims]]

        for val, deriv in pairs:
            if not is_const(val) and not self.cfg.hasopt(cfgsect, deriv):
                raise ValueError(f'Incorrect {deriv} for {val} expression')

    def _init_prescribed(self, cfgsect, subs):
        self._validate_prescribed_cfg(cfgsect)

        params = nirf_src_params(self.ndims)
        self._tplargs = {_to_tplkey(p): self.cfg.getexpr(cfgsect, p, '0.0', subs=subs)
                         for p in params}
        self._tplargs |= nirf_origin_tplargs(self.cfg, cfgsect, self.ndims)

    def _parse_dof(self, cfgsect):
        if self.ndims == 2:
            all_dof = {'x', 'y', 'rz'}
        else:
            all_dof = {'x', 'y', 'z', 'rx', 'ry', 'rz'}

        dof_str = self.cfg.get(cfgsect, 'dof', None)
        if dof_str is None:
            self._free_dof = all_dof
        else:
            self._free_dof = {s.strip() for s in dof_str.split(',')}
            invalid = self._free_dof - all_dof
            if invalid:
                raise ValueError(f"Invalid DOF: {invalid}. Valid: {all_dof}")

        comps = 'xyz'[:self.ndims]
        self._trans_mask = np.array([c in self._free_dof for c in comps])
        self._rot_mask = np.array([f'r{c}' in self._free_dof for c in 'xyz'])

    def _init_free(self, intg, cfgsect, **kwargs):
        comm, rank, root = get_comm_rank_root()

        self._parse_dof(cfgsect)

        self._mass = self.cfg.getfloat(cfgsect, 'mass')
        if self.ndims == 2:
            self._inertia = self.cfg.getfloat(cfgsect, 'inertia')
        else:
            self._inertia = np.array(
                self.cfg.getliteral(cfgsect, 'inertia')).reshape(3, 3)

        comps = 'xyz'[:self.ndims]
        self._fx0 = np.array([
            self.cfg.getfloat(cfgsect, f'frame-origin-{c}')
            for c in comps])

        self._bcname = self.cfg.get(cfgsect, 'boundary')
        self._viscous = 'navier-stokes' in intg.system.name

        if self._viscous:
            self._constants = self.cfg.items_as('constants', float)
            self._viscorr = self.cfg.get('solver', 'viscosity-correction',
                                         'none')

        self._elementscls = intg.system.elementscls

        comps = 'xyz'[:self.ndims]
        if intg.isrestart and kwargs:
            self._fomega = np.array(kwargs['fomega'])
            self._falpha = np.array(kwargs['falpha'])
            self._ftheta = np.array(kwargs.get('ftheta', np.zeros(3)))
            self._fvelo = np.array(kwargs['fvelo'])
            self._faccel = np.array(kwargs['faccel'])
            self._fdx = np.array(kwargs.get('fdx', np.zeros(self.ndims)))
        else:
            self._fomega = np.array([
                self.cfg.getfloat(cfgsect, f'frame-omega0-{c}', 0.0)
                for c in 'xyz'])
            self._ftheta = np.array([
                self.cfg.getfloat(cfgsect, f'frame-theta0-{c}', 0.0)
                for c in 'xyz'])
            self._falpha = np.zeros(3)
            self._fvelo = np.array([
                self.cfg.getfloat(cfgsect, f'frame-velo0-{c}', 0.0)
                for c in comps])
            self._fdx = np.array([
                self.cfg.getfloat(cfgsect, f'frame-dx0-{c}', 0.0)
                for c in comps])
            self._faccel = np.zeros(self.ndims)

        bcranks = comm.gather(self._bcname in intg.system.mesh.bcon, root=root)
        if rank == root and not any(bcranks):
            raise RuntimeError(f'Boundary {self._bcname} does not exist')

        self._ff_int = NIRFForceIntegrator(
            self.cfg, cfgsect, intg.system, self._bcname, self._fx0,
            self._viscous)

        if self.cfg.hasopt(cfgsect, 'dt-ode'):
            self.dt_ode = self.cfg.getfloat(cfgsect, 'dt-ode')
        else:
            self.dt_ode = None
        self.tode_last = intg.tcurr
        self._ode_nout = self.cfg.getint(cfgsect, 'ode-nout', 1)
        self._ode_count = 0

        if self.dt_ode is not None:
            intg.call_plugin_dt(intg.tcurr, self.dt_ode)

        if rank == root and self.cfg.hasopt(cfgsect, 'file'):
            if self.ndims == 2:
                header = ('t,theta,omega,omega_dot,'
                          'dx,dy,accel_x,accel_y,'
                          'velo_x,velo_y,'
                          'fx,fy,mz')
            else:
                header = ('t,theta_x,theta_y,theta_z,'
                          'omega_x,omega_y,omega_z,'
                          'omega_dot_x,omega_dot_y,omega_dot_z,'
                          'dx,dy,dz,accel_x,accel_y,accel_z,'
                          'velo_x,velo_y,velo_z,'
                          'fx,fy,fz,mx,my,mz')
            self._csv = init_csv(self.cfg, cfgsect, header)
        else:
            self._csv = None

        params = nirf_src_params(self.ndims)
        self._tplargs = {_to_tplkey(p): _to_extern(p) for p in params}
        self._tplargs |= nirf_origin_tplargs(self.cfg, cfgsect, self.ndims)

        self._register_externs(intg, [_to_extern(p) for p in params])

    def _update_extern_values(self):
        comps = 'xyz'[:self.ndims]
        ev = self._extern_values

        for i, c in enumerate('xyz'):
            ev[f'frame_omega_{c}'] = self._fomega[i]
            ev[f'frame_alpha_{c}'] = self._falpha[i]

        for i, c in enumerate(comps):
            ev[f'frame_accel_{c}'] = self._faccel[i]
            ev[f'frame_velo_{c}'] = self._fvelo[i]

        # R(-θ) = R(θ)^T
        angle = np.linalg.norm(self._ftheta)
        if angle < 1e-14:
            R = np.eye(3)
        else:
            k = self._ftheta / angle
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)

        # R(-θ) = R(θ)^T for inertial-to-body transform
        Rinv = R.T
        for i in range(3):
            for j in range(3):
                ev[f'nirf_R{i}{j}'] = float(Rinv[i, j])

    def _compute_forces(self, intg):
        comm, rank, root = get_comm_rank_root()

        ndims, nvars = self.ndims, self.nvars
        mcomp = 3 if ndims == 3 else 1

        solns = dict(zip(intg.system.ele_types, intg.soln))
        fm = np.zeros((2 if self._viscous else 1, ndims + mcomp))

        for (etype, fidx), m0 in self._ff_int.m0.items():
            nfpts, nupts = m0.shape

            uupts = solns[etype][..., self._ff_int.eidxs[etype, fidx]]

            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1).swapaxes(0, 1)

            p = self._elementscls.con_to_pri(ufpts, self.cfg)[-1]

            qwts = self._ff_int.qwts[etype, fidx]
            norms = self._ff_int.norms[etype, fidx]

            # Pressure force and moment
            fm[0, :ndims] += np.einsum('i...,ij,jik', qwts, p, norms)

            rfpts = self._ff_int.rfpts[etype, fidx]
            rcn = np.atleast_3d(np.cross(rfpts, norms))
            fm[0, ndims:] += np.einsum('i...,ij,jik->k', qwts, p, rcn)

            if self._viscous:
                m4 = self._ff_int.m4[etype]
                rcpjact = self._ff_int.rcpjact[etype, fidx]

                tduupts = m4 @ uupts.reshape(nupts, -1)
                tduupts = tduupts.reshape(ndims, nupts, nvars, -1)

                duupts = np.einsum('ijkl,jkml->ikml', rcpjact, tduupts)
                duupts = duupts.reshape(ndims, nupts, -1)

                dufpts = np.array([m0 @ du for du in duupts])
                dufpts = dufpts.reshape(ndims, nfpts, nvars, -1).swapaxes(1, 2)

                vis = self._stress_tensor(ufpts, dufpts)

                # Viscous force and moment
                fm[1, :ndims] += np.einsum('i...,klij,jil', qwts, vis, norms)

                viscf = np.einsum('ijkl,lkj->lki', vis, norms)
                rcf = np.atleast_3d(np.cross(rfpts, viscf))
                fm[1, ndims:] += np.einsum('i,jik->k', qwts, rcf)

        if rank != root:
            comm.Reduce(fm, None, op=mpi.SUM, root=root)
        else:
            comm.Reduce(mpi.IN_PLACE, fm, op=mpi.SUM, root=root)

        totals = fm.sum(axis=0) if rank == root else np.zeros(ndims + mcomp)
        comm.Bcast(totals, root=root)

        return totals[:ndims], totals[ndims:]

    def _stress_tensor(self, u, du):
        c = self._constants
        ndims = self.ndims

        rho, E = u[0], u[-1]
        gradrho, gradrhou = du[:, 0], du[:, 1:-1]

        gradu = (gradrhou - gradrho[:, None]*u[None, 1:-1]/rho) / rho
        bulk = np.eye(ndims)[:, :, None, None]*np.trace(gradu)

        mu = c['mu']

        if self._viscorr == 'sutherland':
            cpT = c['gamma']*(E/rho - 0.5*np.sum(u[1:-1]**2, axis=0)/rho**2)
            Trat = cpT/c['cpTref']
            mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

        return -mu*(gradu + gradu.swapaxes(0, 1) - 2/3*bulk)

    def _solve_rigid_body_ode(self, force, moment, dt):
        faccel_new = force / self._mass
        faccel_new *= self._trans_mask

        falpha_new = np.zeros(3)
        if self.ndims == 2:
            falpha_new[2] = moment[0] / self._inertia
        else:
            gyro = np.cross(self._fomega, self._inertia @ self._fomega)
            falpha_new = np.linalg.solve(self._inertia, moment - gyro)
        falpha_new *= self._rot_mask

        # Heun's method
        fomega_new = self._fomega + 0.5*dt*(self._falpha + falpha_new)
        fvelo_new = self._fvelo + 0.5*dt*(self._faccel + faccel_new)

        self._ftheta += 0.5*dt*(self._fomega + fomega_new)
        self._fdx += 0.5*dt*(self._fvelo + fvelo_new)

        self._fomega = fomega_new
        self._falpha = falpha_new
        self._fvelo = fvelo_new
        self._faccel = faccel_new

    def __call__(self, intg):
        if self._motion != 'free':
            return

        if self.dt_ode is not None:
            if intg.tcurr - self.tode_last < self.dt_ode - self.tol:
                return

        force, moment = self._compute_forces(intg)
        ode_dt = intg.tcurr - self.tode_last if self.dt_ode else intg._dt
        self._solve_rigid_body_ode(force, moment, ode_dt)
        self.tode_last = intg.tcurr
        self._bind_externs()

        self._ode_count += 1
        if self._csv and self._ode_count % self._ode_nout == 0:
            _, rank, root = get_comm_rank_root()
            if rank == root:
                if self.ndims == 2:
                    self._csv(intg.tcurr, self._ftheta[2],
                              self._fomega[2], self._falpha[2],
                              self._fdx[0], self._fdx[1],
                              self._faccel[0], self._faccel[1],
                              self._fvelo[0], self._fvelo[1],
                              force[0], force[1], moment[0])
                else:
                    self._csv(intg.tcurr,
                              self._ftheta[0], self._ftheta[1], self._ftheta[2],
                              self._fomega[0], self._fomega[1], self._fomega[2],
                              self._falpha[0], self._falpha[1], self._falpha[2],
                              self._fdx[0], self._fdx[1], self._fdx[2],
                              self._faccel[0], self._faccel[1], self._faccel[2],
                              self._fvelo[0], self._fvelo[1], self._fvelo[2],
                              force[0], force[1], force[2],
                              moment[0], moment[1], moment[2])

    def setup(self, sdata, serialiser):
        if self._motion != 'free':
            return

        if sdata is not None:
            sdata = np.asarray(sdata, dtype=float)
            ndims = self.ndims
            self._fomega = sdata[0:3].copy()
            self._falpha = sdata[3:6].copy()
            self._ftheta = sdata[6:9].copy()
            self._fvelo = sdata[9:9 + ndims].copy()
            self._faccel = sdata[9 + ndims:9 + 2*ndims].copy()
            self._fdx = sdata[9 + 2*ndims:9 + 3*ndims].copy()
            self._update_extern_values()

        serialiser.register(self.get_serialiser_prefix(),
                            self._serialise_data)

    def _serialise_data(self):
        return np.concatenate([self._fomega, self._falpha, self._ftheta,
                               self._fvelo, self._faccel, self._fdx])
