import math

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolverPlugin, init_csv
from pyfr.quadrules.surface import SurfaceIntegrator

# TODO: Add output options for prescribed
# TODO: viscous stress not just for nav-stokes but only no-slp
# TODO: rename frame-origin

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


def _euler_to_quat(phi, theta, psi):
    cp, sp = np.cos(phi / 2), np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cs, ss = np.cos(psi / 2), np.sin(psi / 2)

    return np.array([cp*ct*cs + sp*st*ss,
                     cp*ct*ss - sp*st*cs,
                     cp*st*cs + sp*ct*ss,
                     sp*ct*cs - cp*st*ss])


def _quat_to_euler(q):
    w, x, y, z = q
    phi = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    theta = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    psi = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))

    return phi, theta, psi


def _quat_to_rotmat(q):
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def _quat_mult(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r

    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


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

    def _parse_rot0(self, cfgsect):
        has_euler = self.cfg.hasopt(cfgsect, 'frame-rot0-euler')
        has_quat = self.cfg.hasopt(cfgsect, 'frame-rot0-quat')

        if has_euler and has_quat:
            raise ValueError('Specify frame-rot0-euler or frame-rot0-quat, '
                             'not both')

        if has_euler:
            rot = self.cfg.getliteral(cfgsect, 'frame-rot0-euler')
            if self.ndims == 2:
                phi = np.deg2rad(float(rot))
                return _euler_to_quat(phi, 0, 0)
            else:
                phi, theta, psi = [np.deg2rad(a) for a in rot]
                return _euler_to_quat(phi, theta, psi)
        elif has_quat:
            q = np.array(self.cfg.getliteral(cfgsect, 'frame-rot0-quat'),
                         dtype=float)
            q /= np.linalg.norm(q)
            return q
        else:
            return np.array([1.0, 0.0, 0.0, 0.0])

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

    def _init_free(self, intg, cfgsect):
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

        zeros_nd = (0.,) * self.ndims

        self._fdx = np.array(self.cfg.getliteral(cfgsect, 'frame-dx0', zeros_nd), dtype=float)
        self._fvelo = np.array(self.cfg.getliteral(cfgsect, 'frame-velo0', zeros_nd), dtype=float)
        self._faccel = np.array(self.cfg.getliteral(cfgsect, 'frame-accel0', zeros_nd), dtype=float)
        self._fquat = self._parse_rot0(cfgsect)

        omega0 = self.cfg.getliteral(cfgsect, 'frame-omega0',
                                     0. if self.ndims == 2 else (0., 0., 0.))
        alpha0 = self.cfg.getliteral(cfgsect, 'frame-alpha0',
                                     0. if self.ndims == 2 else (0., 0., 0.))

        if self.ndims == 2:
            if not np.isscalar(omega0) or not np.isscalar(alpha0):
                raise ValueError('frame-omega0/alpha0 must be a scalar in 2D')
            self._fomega = np.array([0., 0., omega0])
            self._falpha = np.array([0., 0., alpha0])
        else:
            self._fomega = np.array(omega0, dtype=float)
            self._falpha = np.array(alpha0, dtype=float)

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
                header = ('t,phi,omega,omega_dot,'
                          'dx,dy,accel_x,accel_y,'
                          'velo_x,velo_y,'
                          'fx,fy,mz')
            else:
                header = ('t,phi,theta,psi,'
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

        # Create a broadcast matrix for R(-θ) and inject into NIRF BC inters
        self._nirf_R = intg.backend.matrix((3, 3), initval=np.eye(3))
        for bc in intg.system._bc_inters:
            # HACK: Need better identifier
            if any(c.__name__ == 'NIRFBCMixin' for c in type(bc).__mro__):
                bc.set_external('nirf_R', 'in broadcast fpdtype_t[3][3]',
                                value=self._nirf_R)

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

        # R(-θ) = R(θ)^T for inertial-to-body transform
        R = _quat_to_rotmat(self._fquat)
        self._nirf_R.set(R.T)

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
            Trat = np.maximum(cpT/c['cpTref'], 1e-10)
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

        omega_avg = 0.5*(self._fomega + fomega_new)
        dqdt = 0.5*_quat_mult(self._fquat, np.array([0, *omega_avg]))
        self._fquat += dt*dqdt
        self._fquat /= np.linalg.norm(self._fquat)

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
        self._update_extern_values()
        self.bind_externs()

        self._ode_count += 1
        if self._csv and self._ode_count % self._ode_nout == 0:
            _, rank, root = get_comm_rank_root()
            if rank == root:
                phi, theta, psi = _quat_to_euler(self._fquat)
                phi, theta, psi = (np.rad2deg(phi), np.rad2deg(theta),
                                   np.rad2deg(psi))

                if self.ndims == 2:
                    self._csv(intg.tcurr, phi,
                              self._fomega[2], self._falpha[2],
                              self._fdx[0], self._fdx[1],
                              self._faccel[0], self._faccel[1],
                              self._fvelo[0], self._fvelo[1],
                              force[0], force[1], moment[0])
                else:
                    self._csv(intg.tcurr,
                              phi, theta, psi,
                              self._fomega[0], self._fomega[1], self._fomega[2],
                              self._falpha[0], self._falpha[1], self._falpha[2],
                              self._fdx[0], self._fdx[1], self._fdx[2],
                              self._faccel[0], self._faccel[1], self._faccel[2],
                              self._fvelo[0], self._fvelo[1], self._fvelo[2],
                              force[0], force[1], force[2],
                              moment[0], moment[1], moment[2])

    _sdata_dtype = np.dtype([
        ('dx', 'f8', 3), ('velo', 'f8', 3), ('accel', 'f8', 3),
        ('quat', 'f8', 4), ('omega', 'f8', 3), ('alpha', 'f8', 3),
        ('tode_last', 'f8')
    ])

    def setup(self, sdata, serialiser):
        if self._motion != 'free':
            return

        if sdata is not None:
            ndims = self.ndims
            self._fdx = np.array(sdata['dx'])[:ndims].copy()
            self._fvelo = np.array(sdata['velo'])[:ndims].copy()
            self._faccel = np.array(sdata['accel'])[:ndims].copy()
            self._fquat = np.array(sdata['quat'])
            self._fomega = np.array(sdata['omega'])
            self._falpha = np.array(sdata['alpha'])
            self.tode_last = float(sdata['tode_last'])
            self._update_extern_values()

        serialiser.register(self.get_serialiser_prefix(),
                            self._serialise_data)

    def _serialise_data(self):
        pad = 3 - self.ndims
        return np.void((
            np.pad(self._fdx, (0, pad)),
            np.pad(self._fvelo, (0, pad)),
            np.pad(self._faccel, (0, pad)),
            self._fquat, self._fomega, self._falpha,
            self.tode_last
        ), dtype=self._sdata_dtype)
