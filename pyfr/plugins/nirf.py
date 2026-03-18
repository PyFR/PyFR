"""Non-Inertial Reference Frame (NIRF) plugin.

Simulates flow in a non-inertial (accelerating/rotating) reference frame by
adding fictitious body-force source terms to the governing equations.  Two
modes are available:

  prescribed — frame motion is specified analytically as expressions in t.
  free       — frame motion is computed by integrating the rigid-body ODE
               driven by aerodynamic forces on a designated boundary.

Both modes serialise kinematic state (location, velocity, orientation,
angular velocity) on checkpoint, so a prescribed run can be restarted in free
mode without any loss of continuity.

Config section: [solver-plugin-nirf]
-------------------------------------
Common options (both modes)
  motion           str    'prescribed' or 'free'.  Default: 'prescribed'.
  center-of-rot    tuple  Centre of rotation / pivot point, length-ndims float
                          tuple, e.g. (0.0, 0.5).  Default: (0,)*ndims.

Prescribed-mode options
  All motion parameters can be expressions of t.  The user must provide the
  complete kinematic stack: loc → velo → accel for translation, and
  rot → omega → alpha for rotation.

  frame-loc-{x,y[,z]}      expr  Frame position (m).  Default: 0.0.
  frame-velo-{x,y[,z]}     expr  Translational velocity (m/s).  Default: 0.0.
  frame-accel-{x,y[,z]}    expr  Translational acceleration (m/s²).
                                  Default: 0.0.
  frame-rot-z               expr  2-D: rotation angle about z (rad).
                                  Default: 0.0.
  frame-rot-{x,y,z}        expr  3-D: ZYX Euler angles (rad).  Default: 0.0.
  frame-omega-{x,y,z}      expr  Angular velocity (rad/s).  Default: 0.0.
  frame-alpha-{x,y,z}      expr  Angular acceleration (rad/s²).  Default: 0.0.

Free-mode options
  mass             float  Body mass.  Required.
  inertia          float  2-D: scalar moment of inertia about z.
                   tuple  3-D: 3×3 inertia tensor as a flat 9-element tuple.
                          Required.
  boundary         str    Name of the body-surface boundary for force
                          integration.  Required.
  dof              str    Comma-separated active degrees of freedom.
                          2-D: any subset of {x, y, rz}.
                          3-D: any subset of {x, y, z, rx, ry, rz}.
                          Default: all DOF for the current dimensionality.

  Initial conditions:
  frame-loc0       tuple  Initial frame position (m).  Default: (0,)*ndims.
  frame-velo0      tuple  Initial translational velocity (m/s).
                          Default: (0,)*ndims.
  frame-accel0     tuple  Initial translational acceleration (m/s²).
                          Default: (0,)*ndims.
  frame-rot0-euler float  2-D: initial rotation angle in radians.
                   tuple  3-D: (phi, theta, psi) ZYX Euler angles in radians.
                          Mutually exclusive with frame-rot0-quat.
  frame-rot0-quat  tuple  Initial orientation as (w, x, y, z) quaternion.
                          Automatically normalised.
                          Mutually exclusive with frame-rot0-euler.
  frame-omega0     float  2-D: initial angular velocity (rad/s) about z.
                   tuple  3-D: (wx, wy, wz).  Default: 0.0 / (0, 0, 0).
  frame-alpha0     float  2-D: initial angular acceleration (rad/s²) about z.
                   tuple  3-D: (ax, ay, az).  Default: 0.0 / (0, 0, 0).

  Output options (free mode only):
  ode-nout         int    Write CSV output every N ODE steps.  Default: 1.
  dt-ode           float  ODE sub-step size (s).  Default: solver dt.
  file             str    Path for CSV output.  Optional.

"""

import math

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolverPlugin, init_csv
from pyfr.quadrules.surface import SurfaceIntegrator

# TODO: viscous stress not just for nav-stokes but only no-slp
# TODO: add support for multiple boundary names

# Python-side math namespace for evaluating getexpr strings
_PYMATH = {name: getattr(math, name)
           for name in dir(math)
           if not name.startswith('_')}


def _eval_expr(expr, t):
    return eval(expr, _PYMATH, {'t': t})


def nirf_src_params(ndims):
    comps = 'xyz'[:ndims]
    return ([f'frame-omega-{c}' for c in 'xyz'] +
            [f'frame-alpha-{c}' for c in 'xyz'] +
            [f'frame-accel-{c}' for c in comps])


def nirf_bc_params(ndims):
    comps = 'xyz'[:ndims]
    return ([f'frame-omega-{c}' for c in 'xyz'] +
            [f'frame-loc-{c}' for c in comps] +
            [f'frame-velo-{c}' for c in comps])


def nirf_origin_tplargs(cfg, cfgsect, ndims):
    origin = cfg.getliteral(cfgsect, 'center-of-rot', (0.,) * ndims)
    return {f'frame_origin_{c}': v for c, v in zip('xyz'[:ndims], origin)}


def _to_tplkey(p):
    return p.replace('-', '_') + '_expr'


def _to_extern(p):
    return p.replace('-', '_')


# ZYX intrinsic: phi=Z, theta=Y, psi=X
def _euler_to_quat(phi, theta, psi):
    cp, sp = np.cos(phi / 2), np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cs, ss = np.cos(psi / 2), np.sin(psi / 2)

    return np.array([cp*ct*cs + sp*st*ss, cp*ct*ss - sp*st*cs,
                     cp*st*cs + sp*ct*ss, sp*ct*cs - cp*st*ss])


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
            self._call = self._call_prescribed
        elif self._motion == 'free':
            self._init_free(intg, cfgsect)
            self._call = self._call_free
        else:
            raise ValueError(
                f"Invalid NIRF motion type: {self._motion}"
            )

        # Transform ICs from lab frame to body frame (fresh start)
        if not intg.isrestart:
            self._transform_ics(intg, cfgsect)

        macro = f'nirf_source_{self.ndims}d'
        for eles in intg.system.ele_map.values():
            eles.add_src_macro('pyfr.plugins.kernels.nirf', macro,
                               self._tplargs, ploc=True, soln=True)

    def _validate_prescribed_cfg(self, cfgsect, subs):
        comps = 'xyz'[:self.ndims]
        ge = self.cfg.getexpr

        def ev(exprs, t):
            return np.array([_eval_expr(e, t) for e in exprs])

        def _rich(f, t, h):
            fp = f(t + h)
            fm = f(t - h)
            fph = f(t + h / 2)
            fmh = f(t - h / 2)
            d1 = (fp - fm) / (2 * h)
            d2 = (fph - fmh) / h
            return (4 * d2 - d1) / 3

        def converged_fd(f, t):
            h = 1e-4
            prev = _rich(f, t, h)
            for _ in range(20):
                h /= 2
                curr = _rich(f, t, h)
                scale = max(np.max(np.abs(curr)), 1.0)
                if np.max(np.abs(curr - prev)) / scale < 1e-8:
                    return curr
                prev = curr
            return curr

        def check(dname, fname, fd, given):
            denom = max(np.max(np.abs(given)),
                        np.max(np.abs(fd)), 1.0)
            err = np.max(np.abs(fd - given)) / denom
            if err > 1e-6:
                raise ValueError(
                    f'{dname} is not the derivative '
                    f'of {fname}; numerical={fd}, '
                    f'specified={given}'
                )

        s = subs
        loc = [ge(cfgsect, f'frame-loc-{c}', '0.0', subs=s)
               for c in comps]
        velo = [ge(cfgsect, f'frame-velo-{c}', '0.0', subs=s)
                for c in comps]
        accel = [ge(cfgsect, f'frame-accel-{c}', '0.0', subs=s)
                 for c in comps]
        omega = [ge(cfgsect, f'frame-omega-{c}', '0.0', subs=s)
                 for c in 'xyz']
        alpha = [ge(cfgsect, f'frame-alpha-{c}', '0.0', subs=s)
                 for c in 'xyz']

        if self.ndims == 2:
            rot = [ge(cfgsect, 'frame-rot-z', '0.0', subs=s)]
        else:
            rot = [ge(cfgsect, f'frame-rot-{c}', '0.0', subs=s)
                   for c in 'xyz']

        t = 1.0

        # loc -> velo -> accel
        fd_velo = converged_fd(lambda s: ev(loc, s), t)
        check('frame-velo', 'frame-loc',
              fd_velo, ev(velo, t))
        fd_accel = converged_fd(lambda s: ev(velo, s), t)
        check('frame-accel', 'frame-velo',
              fd_accel, ev(accel, t))

        # rot -> omega via quaternion kinematics
        def quat_at(t):
            if self.ndims == 2:
                phi = _eval_expr(rot[0], t)
                return _euler_to_quat(phi, 0, 0)
            return _euler_to_quat(*ev(rot, t)[::-1])

        dqdt_fd = converged_fd(quat_at, t)
        q0 = quat_at(t)
        w = ev(omega, t)
        dqdt_an = 0.5 * _quat_mult(q0, np.r_[0, w])
        check('frame-omega', 'frame-rot',
              dqdt_fd, dqdt_an)

        # omega -> alpha
        fd_alpha = converged_fd(lambda s: ev(omega, s), t)
        check('frame-alpha', 'frame-omega',
              fd_alpha, ev(alpha, t))

    def _transform_ics(self, intg, cfgsect):
        ndims = self.ndims
        cor = np.array(self.cfg.getliteral(
            cfgsect, 'center-of-rot', (0.,) * ndims
        ))

        Rt = _quat_to_rotmat(self._fquat).T
        omega = self._fomega
        velo = self._fvelo

        solns = intg.system.ele_scal_upts(0)
        for s, ploc in zip(solns, intg.system.ele_ploc_upts):
            rho = s[:, 0, :]
            u_lab = s[:, 1:1 + ndims, :] / rho[:, None, :]

            # Pad to 3D for cross product and rotation
            nupts, neles = ploc.shape[0], ploc.shape[-1]
            r3 = np.zeros((nupts, 3, neles))
            r3[:, :ndims, :] = ploc - cor[None, :, None]

            u3 = np.zeros_like(r3)
            u3[:, :ndims, :] = u_lab - velo[None, :, None]

            # u_body = R^T*(u_lab - V_frame) - Ω×r
            ub = np.einsum('ij,ajk->aik', Rt, u3)
            ub -= np.cross(omega, r3.transpose(0, 2, 1)).transpose(0, 2, 1)

            # Update momentum and correct energy for KE change
            ke_old = 0.5 * rho * np.sum(u_lab**2, axis=1)
            ke_new = 0.5 * rho * np.sum(ub[:, :ndims, :]**2, axis=1)
            s[:, 1:1 + ndims, :] = rho[:, None, :] * ub[:, :ndims, :]
            s[:, -1, :] += ke_new - ke_old

        # Write transformed solution back to backend
        for eb, s in zip(intg.system.ele_banks, solns):
            eb[0].set(s)

    def _init_prescribed(self, cfgsect, subs):
        self._validate_prescribed_cfg(cfgsect, subs)

        comps = 'xyz'[:self.ndims]

        # Source term params (omega, alpha, accel)
        params = nirf_src_params(self.ndims)
        self._tplargs = {_to_tplkey(p): self.cfg.getexpr(cfgsect, p, '0.0', subs=subs)
                         for p in params}
        self._tplargs |= nirf_origin_tplargs(self.cfg, cfgsect, self.ndims)

        # Prescribed expressions for direct evaluation
        self._loc_exprs = [self.cfg.getexpr(cfgsect, f'frame-loc-{c}', '0.0', subs=subs)
                           for c in comps]
        self._velo_exprs = [self.cfg.getexpr(cfgsect, f'frame-velo-{c}', '0.0', subs=subs)
                            for c in comps]
        if self.ndims == 2:
            self._rot_exprs = [self.cfg.getexpr(cfgsect, 'frame-rot-z', '0.0', subs=subs)]
        else:
            self._rot_exprs = [self.cfg.getexpr(cfgsect, f'frame-rot-{c}', '0.0', subs=subs)
                               for c in 'xyz']

        self._eval_prescribed(self._intg.tcurr)
        self._init_nirf_R()

    def _parse_rot0(self, cfgsect):
        has_euler = self.cfg.hasopt(cfgsect, 'frame-rot0-euler')
        has_quat = self.cfg.hasopt(cfgsect, 'frame-rot0-quat')

        if has_euler and has_quat:
            raise ValueError('Specify frame-rot0-euler or frame-rot0-quat, '
                             'not both')

        if has_euler:
            rot = self.cfg.getliteral(cfgsect, 'frame-rot0-euler')
            if self.ndims == 2:
                return _euler_to_quat(float(rot), 0, 0)
            else:
                return _euler_to_quat(*rot[::-1])
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
        elif dof_str.strip().lower() in ('', 'none'):
            self._free_dof = set()
        else:
            self._free_dof = {s.strip() for s in dof_str.split(',')}
            invalid = self._free_dof - all_dof
            if invalid:
                raise ValueError(f"Invalid DOF: {invalid}. Valid: {all_dof}")

        comps = 'xyz'[:self.ndims]
        self._trans_mask = np.array([c in self._free_dof for c in comps])
        self._rot_mask = np.array([f'r{c}' in self._free_dof for c in 'xyz'])

    def _init_force_integrator(self, cfgsect):
        intg = self._intg
        comm, rank, root = get_comm_rank_root()

        self._bcname = self.cfg.get(cfgsect, 'boundary')
        self._viscous = 'navier-stokes' in intg.system.name

        if self._viscous:
            self._constants = self.cfg.items_as('constants', float)
            self._viscorr = self.cfg.get('solver', 'viscosity-correction',
                                         'none')

        self._elementscls = intg.system.elementscls

        fx0 = np.array(self.cfg.getliteral(cfgsect, 'center-of-rot',
                                           (0.,) * self.ndims), dtype=float)

        bcranks = comm.gather(self._bcname in intg.system.mesh.bcon, root=root)
        if rank == root and not any(bcranks):
            raise RuntimeError(f'Boundary {self._bcname} does not exist')

        self._ff_int = NIRFForceIntegrator(
            self.cfg, cfgsect, intg.system, self._bcname, fx0, self._viscous)

    def _init_free(self, intg, cfgsect):
        self._parse_dof(cfgsect)

        self._mass = self.cfg.getfloat(cfgsect, 'mass')
        if self.ndims == 2:
            self._inertia = self.cfg.getfloat(cfgsect, 'inertia')
        else:
            self._inertia = np.array(
                self.cfg.getliteral(cfgsect, 'inertia')).reshape(3, 3)

        zeros_nd = (0.,) * self.ndims

        self._floc = np.array(self.cfg.getliteral(cfgsect, 'frame-loc0', zeros_nd), dtype=float)
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

        self._init_force_integrator(cfgsect)

        if self.cfg.hasopt(cfgsect, 'dt-ode'):
            self.dt_ode = self.cfg.getfloat(cfgsect, 'dt-ode')
        else:
            self.dt_ode = None
        self.tode_last = intg.tcurr

        if self.dt_ode is not None:
            intg.call_plugin_dt(intg.tcurr, self.dt_ode)

        params = nirf_src_params(self.ndims)
        self._tplargs = {_to_tplkey(p): _to_extern(p) for p in params}
        self._tplargs |= nirf_origin_tplargs(self.cfg, cfgsect, self.ndims)

        self._init_nirf_R()
        self._register_externs(intg, [_to_extern(p) for p in params])

        _, rank, root = get_comm_rank_root()
        self._ode_nout = self.cfg.getint(cfgsect, 'ode-nout', 1)
        self._ode_count = 0

        if rank == root and self.cfg.hasopt(cfgsect, 'file'):
            if self.ndims == 2:
                header = ('t,phi,omega,omega_dot,loc_x,loc_y,'
                          'velo_x,velo_y,accel_x,accel_y,fx,fy,mz')
            else:
                header = ('t,phi,theta,psi,'
                          'omega_x,omega_y,omega_z,'
                          'omega_dot_x,omega_dot_y,omega_dot_z,'
                          'loc_x,loc_y,loc_z,'
                          'velo_x,velo_y,velo_z,'
                          'accel_x,accel_y,accel_z,'
                          'fx,fy,fz,mx,my,mz')
            self._csv = init_csv(self.cfg, cfgsect, header)
        else:
            self._csv = None

    def _update_extern_values(self):
        comps = 'xyz'[:self.ndims]
        ev = self._extern_values

        for i, c in enumerate('xyz'):
            ev[f'frame_omega_{c}'] = self._fomega[i]
            ev[f'frame_alpha_{c}'] = self._falpha[i]

        for i, c in enumerate(comps):
            ev[f'frame_loc_{c}'] = self._floc[i]
            ev[f'frame_velo_{c}'] = self._fvelo[i]
            ev[f'frame_accel_{c}'] = self._faccel[i]

        self._update_nirf_R()

    def _init_nirf_R(self):
        intg = self._intg
        R0 = _quat_to_rotmat(self._fquat).T
        self._nirf_R = intg.backend.matrix(
            (3, 3), initval=R0
        )
        for bc in intg.system._bc_inters:
            if any(c.__name__ == 'NIRFBCMixin'
                   for c in type(bc).__mro__):
                bc.set_external(
                    'nirf_R',
                    'in broadcast fpdtype_t[3][3]',
                    value=self._nirf_R
                )

    def _update_nirf_R(self):
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

        self._floc += 0.5*dt*(self._fvelo + fvelo_new)

        self._fomega = fomega_new
        self._falpha = falpha_new
        self._fvelo = fvelo_new
        self._faccel = faccel_new

    def __call__(self, intg):
        self._call(intg)
        self._update_nirf_R()

    def _eval_prescribed(self, t):
        comps = 'xyz'[:self.ndims]
        self._floc = np.array([_eval_expr(e, t) for e in self._loc_exprs])
        self._fvelo = np.array([_eval_expr(e, t) for e in self._velo_exprs])
        self._faccel = np.array([_eval_expr(self._tplargs[_to_tplkey(f'frame-accel-{c}')], t)
                                 for c in comps])
        self._fomega = np.array([_eval_expr(self._tplargs[_to_tplkey(f'frame-omega-{c}')], t)
                                 for c in 'xyz'])
        self._falpha = np.array([_eval_expr(self._tplargs[_to_tplkey(f'frame-alpha-{c}')], t)
                                 for c in 'xyz'])
        if self.ndims == 2:
            self._fquat = _euler_to_quat(_eval_expr(self._rot_exprs[0], t), 0, 0)
        else:
            self._fquat = _euler_to_quat(*[_eval_expr(e, t) for e in self._rot_exprs[::-1]])
        self.tode_last = t

    def _call_prescribed(self, intg):
        self._eval_prescribed(intg.tcurr)

    def _call_free(self, intg):
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
        if self._ode_count % self._ode_nout == 0:
            self._write_csv(intg, force, moment)

    def _write_csv(self, intg, force, moment):
        if not self._csv:
            return

        _, rank, root = get_comm_rank_root()
        if rank != root:
            return

        phi, theta, psi = _quat_to_euler(self._fquat)

        if self.ndims == 2:
            args = [intg.tcurr, phi,
                    self._fomega[2], self._falpha[2],
                    self._floc[0], self._floc[1],
                    self._fvelo[0], self._fvelo[1],
                    self._faccel[0], self._faccel[1],
                    force[0], force[1], moment[0]]
        else:
            args = [intg.tcurr, phi, theta, psi,
                    self._fomega[0], self._fomega[1], self._fomega[2],
                    self._falpha[0], self._falpha[1], self._falpha[2],
                    self._floc[0], self._floc[1], self._floc[2],
                    self._fvelo[0], self._fvelo[1], self._fvelo[2],
                    self._faccel[0], self._faccel[1], self._faccel[2],
                    force[0], force[1], force[2],
                    moment[0], moment[1], moment[2]]

        self._csv(*args)

    _sdata_dtype = np.dtype([
        ('loc', 'f8', 3), ('velo', 'f8', 3), ('accel', 'f8', 3),
        ('quat', 'f8', 4), ('omega', 'f8', 3), ('alpha', 'f8', 3),
        ('tode_last', 'f8')
    ])

    def setup(self, sdata, serialiser):
        if sdata is not None:
            ndims = self.ndims
            self._floc = np.array(sdata['loc'])[:ndims].copy()
            self._fvelo = np.array(sdata['velo'])[:ndims].copy()
            self._faccel = np.array(sdata['accel'])[:ndims].copy()
            self._fquat = np.array(sdata['quat'])
            self._fomega = np.array(sdata['omega'])
            self._falpha = np.array(sdata['alpha'])
            self.tode_last = float(sdata['tode_last'])
            if self._motion == 'free':
                self._update_extern_values()

        serialiser.register(self.get_serialiser_prefix(),
                            self._serialise_data)

    def _serialise_data(self):
        pad = 3 - self.ndims
        return np.void((
            np.pad(self._floc, (0, pad)),
            np.pad(self._fvelo, (0, pad)),
            np.pad(self._faccel, (0, pad)),
            self._fquat, self._fomega, self._falpha,
            self.tode_last
        ), dtype=self._sdata_dtype)
