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

Lab-frame export
-----------------
A companion postproc plugin (`nirf`) rotates coordinates and velocity from
the body frame back to the inertial (lab) frame at export time:

  pyfr export volume --postproc nirf mesh.pyfrm in.pyfrs out.vtu

Config section: [postproc-plugin-nirf]
  apply-translation  bool   Add the frame translation `loc` after rotation
                            (body sits at its lab-frame position).
                            Default: False (rotation only, body stays put).

"""

import math

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.exprs import npeval
from pyfr.plugins.common import init_csv
from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.quadrules.surface import SurfaceIntegrator
from pyfr.util import subclass_where

# TODO: viscous stress not just for nav-stokes but only no-slp
# TODO: add support for multiple boundary names

def _eval_expr(expr, t):
    return eval(expr, vars(math), {'t': t})


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
        con = system.mesh.bcon.get(bcname)

        super().__init__(cfg, cfgsect, system.ele_map, con, flags='s')

        self.ndims = system.ndims
        self.viscous = viscous
        self.ele_types = system.ele_types
        self.elementscls = system.elementscls

        if con is not None and morigin is not None:
            self.rfpts = {k: loc - morigin[:, None, None]
                         for k, loc in self.locs.items()}

        # Viscous: stress-model constants plus the per-element gradient
        # operators (m4) and J^-T (rcpjact) used by grad_at_fpts
        if viscous:
            self.constants = cfg.items_as('constants', float)
            self.viscorr = cfg.get('solver', 'viscosity-correction', 'none')

            self.m4 = {}
            rcpjact = {}
            for etype in system.ele_map:
                eles = system.ele_map[etype]

                self.m4[etype] = eles.basis.m4
                smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
                rcpdjac = eles.rcpdjac_at_np('upts')
                rcpjact[etype] = smat * rcpdjac

            # Keep only the boundary elements
            self.rcpjact = {k: rcpjact[k[0]][..., v]
                           for k, v in self.eidxs.items()}

    def grad_at_fpts(self, etype, fidx, uupts):
        # Reconstruct the physical solution gradient at the face quadrature
        # points.  uupts is (s, v, e); the result is (d, v, f, e).
        m0 = self.m0[etype, fidx]
        m4 = self.m4[etype]
        rcpjact = self.rcpjact[etype, fidx]

        nfpts, nupts = m0.shape
        ndims, nvars = self.ndims, uupts.shape[1]

        # Transformed gradient at solution points: (r, s, v, e)
        tdu = (m4 @ uupts.reshape(nupts, -1)).reshape(ndims, nupts, nvars, -1)

        # Map reference -> physical gradient via J^-T: (d, s, v, e)
        du = np.einsum('drse,rsve->dsve', rcpjact, tdu)

        # Interpolate each physical-gradient component to the face points
        dufpts = np.array([m0 @ d for d in du.reshape(ndims, nupts, -1)])
        return dufpts.reshape(ndims, nfpts, nvars, -1).swapaxes(1, 2)

    def compute(self, soln):
        comm, rank, root = get_comm_rank_root()

        ndims = self.ndims
        mcomp = 3 if ndims == 3 else 1

        solns = dict(zip(self.ele_types, soln))
        fm = np.zeros((2 if self.viscous else 1, ndims + mcomp))

        # einsum axes: f face point, e element, d/c spatial dim, m moment comp
        for (etype, fidx), m0 in self.m0.items():
            nfpts, nupts = m0.shape

            uupts = solns[etype][..., self.eidxs[etype, fidx]]
            nvars = uupts.shape[1]

            # Interpolate to face points: (v, f, e)
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1).swapaxes(0, 1)

            p = self.elementscls.con_to_pri(ufpts, self.cfg)[-1]

            qwts = self.qwts[etype, fidx]

            # Reorient normals and moment arms to (f, e, d)
            norms = self.norms[etype, fidx].transpose(1, 2, 0)
            rfpts = self.rfpts[etype, fidx].transpose(1, 2, 0)

            # Pressure force (d) and moment (m): F = ∮ p n, M = ∮ r × (p n)
            fm[0, :ndims] += np.einsum('f,fe,fed->d', qwts, p, norms)

            rcn = np.atleast_3d(np.cross(rfpts, norms))
            fm[0, ndims:] += np.einsum('f,fe,fem->m', qwts, p, rcn)

            if self.viscous:
                # Viscous stress τ at face points, reoriented to (f, e, d, c)
                dufpts = self.grad_at_fpts(etype, fidx, uupts)
                vis = self._stress_tensor(ufpts, dufpts).transpose(2, 3, 0, 1)

                # Viscous force (d) and moment (m): traction t_d = τ_dc n_c
                fm[1, :ndims] += np.einsum('f,fedc,fec->d', qwts, vis, norms)

                viscf = np.einsum('fedc,fec->fed', vis, norms)
                rcf = np.atleast_3d(np.cross(rfpts, viscf))
                fm[1, ndims:] += np.einsum('f,fem->m', qwts, rcf)

        if rank != root:
            comm.Reduce(fm, None, op=mpi.SUM, root=root)
        else:
            comm.Reduce(mpi.IN_PLACE, fm, op=mpi.SUM, root=root)

        totals = fm.sum(axis=0) if rank == root else np.zeros(ndims + mcomp)
        comm.Bcast(totals, root=root)

        return totals[:ndims], totals[ndims:]

    def _stress_tensor(self, u, du):
        c = self.constants
        ndims = self.ndims

        rho, E = u[0], u[-1]
        gradrho, gradrhou = du[:, 0], du[:, 1:-1]

        gradu = (gradrhou - gradrho[:, None]*u[None, 1:-1]/rho) / rho
        bulk = np.eye(ndims)[:, :, None, None]*np.trace(gradu)

        mu = c['mu']

        if self.viscorr == 'sutherland':
            cpT = c['gamma']*(E/rho - 0.5*np.sum(u[1:-1]**2, axis=0)/rho**2)
            Trat = np.maximum(cpT/c['cpTref'], 1e-10)
            mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

        return -mu*(gradu + gradu.swapaxes(0, 1) - 2/3*bulk)


class BaseMotion:
    name = None
    needs_force = False
    has_externs = False
    extern_names = ()

    def __init__(self, plugin, cfgsect):
        self.plugin = plugin
        self.cfg = plugin.cfg
        self.ndims = plugin.ndims
        self.cfgsect = cfgsect

        # State (subclass __init__ populates)
        self.floc = np.zeros(self.ndims)
        self.fvelo = np.zeros(self.ndims)
        self.faccel = np.zeros(self.ndims)
        self.fquat = np.array([1.0, 0.0, 0.0, 0.0])
        self.fomega = np.zeros(3)
        self.falpha = np.zeros(3)
        self.tode_last = plugin._intg.tcurr

        # Kernel template args (subclass __init__ populates)
        self.tplargs = {}

    def should_advance(self, intg):
        return True

    def advance(self, intg, force, moment):
        raise NotImplementedError


class PrescribedMotion(BaseMotion):
    name = 'prescribed'

    def __init__(self, plugin, cfgsect):
        super().__init__(plugin, cfgsect)

        subs = self.cfg.items('constants')
        subs |= dict(abs='fabs', pi=math.pi)

        self.fexprs = self._parse_motion_exprs(subs)
        self._validate_cfg()
        self._build_tplargs()

        self.advance_to(plugin._intg.tcurr)

    def _parse_motion_exprs(self, subs):
        ge = self.cfg.getexpr
        cfgsect = self.cfgsect
        comps = 'xyz'[:self.ndims]

        exprs = {}
        for kind in ('loc', 'velo', 'accel'):
            exprs[kind] = [ge(cfgsect, f'frame-{kind}-{c}', '0.0', subs=subs)
                           for c in comps]
        for kind in ('omega', 'alpha'):
            exprs[kind] = [ge(cfgsect, f'frame-{kind}-{c}', '0.0', subs=subs)
                           for c in 'xyz']
        if self.ndims == 2:
            exprs['rot'] = [ge(cfgsect, 'frame-rot-z', '0.0', subs=subs)]
        else:
            exprs['rot'] = [ge(cfgsect, f'frame-rot-{c}', '0.0', subs=subs)
                            for c in 'xyz']
        return exprs

    def _build_tplargs(self):
        for kind, comps in (('omega', 'xyz'), ('alpha', 'xyz'),
                            ('accel', 'xyz'[:self.ndims])):
            for i, c in enumerate(comps):
                key = _to_tplkey(f'frame-{kind}-{c}')
                self.tplargs[key] = self.fexprs[kind][i]
        self.tplargs |= nirf_origin_tplargs(self.cfg, self.cfgsect, self.ndims)

    def _validate_cfg(self):
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

        t = 1.0

        for deriv, primary in (('velo', 'loc'),
                               ('accel', 'velo'),
                               ('alpha', 'omega')):
            fd = converged_fd(
                lambda s, p=primary: ev(self.fexprs[p], s), t
            )
            check(f'frame-{deriv}', f'frame-{primary}',
                  fd, ev(self.fexprs[deriv], t))

        # rot -> omega via quaternion kinematics
        def quat_at(t):
            rv = ev(self.fexprs['rot'], t)
            if self.ndims == 2:
                return _euler_to_quat(rv[0], 0, 0)
            return _euler_to_quat(*rv[::-1])

        dqdt_fd = converged_fd(quat_at, t)
        w = ev(self.fexprs['omega'], t)
        dqdt_an = 0.5 * _quat_mult(quat_at(t), np.r_[0, w])
        check('frame-omega', 'frame-rot', dqdt_fd, dqdt_an)

    def advance_to(self, t):
        vals = {k: np.array([_eval_expr(e, t) for e in exprs])
                for k, exprs in self.fexprs.items()}

        self.floc = vals['loc']
        self.fvelo = vals['velo']
        self.faccel = vals['accel']
        self.fomega = vals['omega']
        self.falpha = vals['alpha']

        if self.ndims == 2:
            self.fquat = _euler_to_quat(vals['rot'][0], 0, 0)
        else:
            self.fquat = _euler_to_quat(*vals['rot'][::-1])

        self.tode_last = t

    def advance(self, intg, force, moment):
        self.advance_to(intg.tcurr)


class FreeMotion(BaseMotion):
    name = 'free'
    needs_force = True
    has_externs = True

    def __init__(self, plugin, cfgsect):
        super().__init__(plugin, cfgsect)

        self._parse_dof()

        self.mass = self.cfg.getfloat(cfgsect, 'mass')
        if self.ndims == 2:
            self.inertia = self.cfg.getfloat(cfgsect, 'inertia')
        else:
            self.inertia = np.array(
                self.cfg.getliteral(cfgsect, 'inertia')).reshape(3, 3)

        zeros_nd = (0.,) * self.ndims

        self.floc = np.array(self.cfg.getliteral(
            cfgsect, 'frame-loc0', zeros_nd), dtype=float)
        self.fvelo = np.array(self.cfg.getliteral(
            cfgsect, 'frame-velo0', zeros_nd), dtype=float)
        self.faccel = np.array(self.cfg.getliteral(
            cfgsect, 'frame-accel0', zeros_nd), dtype=float)
        self.fquat = self._parse_rot0()

        omega0 = self.cfg.getliteral(cfgsect, 'frame-omega0',
                                     0. if self.ndims == 2 else (0., 0., 0.))
        alpha0 = self.cfg.getliteral(cfgsect, 'frame-alpha0',
                                     0. if self.ndims == 2 else (0., 0., 0.))

        if self.ndims == 2:
            if not np.isscalar(omega0) or not np.isscalar(alpha0):
                raise ValueError('frame-omega0/alpha0 must be a scalar in 2D')
            self.fomega = np.array([0., 0., omega0])
            self.falpha = np.array([0., 0., alpha0])
        else:
            self.fomega = np.array(omega0, dtype=float)
            self.falpha = np.array(alpha0, dtype=float)

        if self.cfg.hasopt(cfgsect, 'dt-ode'):
            self.dt_ode = self.cfg.getfloat(cfgsect, 'dt-ode')
        else:
            self.dt_ode = None

        self.tode_last = plugin._intg.tcurr

        if self.dt_ode is not None:
            plugin._intg.call_plugin_dt(plugin._intg.tcurr, self.dt_ode)

        params = nirf_src_params(self.ndims)
        self.tplargs = {_to_tplkey(p): _to_extern(p) for p in params}
        self.tplargs |= nirf_origin_tplargs(self.cfg, cfgsect, self.ndims)
        self.extern_names = [_to_extern(p) for p in params]

    def _parse_dof(self):
        if self.ndims == 2:
            all_dof = {'x', 'y', 'rz'}
        else:
            all_dof = {'x', 'y', 'z', 'rx', 'ry', 'rz'}

        dof_str = self.cfg.get(self.cfgsect, 'dof', None)
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

    def _parse_rot0(self):
        cfgsect = self.cfgsect
        has_euler = self.cfg.hasopt(cfgsect, 'frame-rot0-euler')
        has_quat = self.cfg.hasopt(cfgsect, 'frame-rot0-quat')

        if has_euler and has_quat:
            raise ValueError('Specify frame-rot0-euler or frame-rot0-quat, '
                             'not both')

        if has_euler:
            rot = self.cfg.getliteral(cfgsect, 'frame-rot0-euler')
            if self.ndims == 2:
                return _euler_to_quat(float(rot), 0, 0)
            return _euler_to_quat(*rot[::-1])
        elif has_quat:
            q = np.array(self.cfg.getliteral(cfgsect, 'frame-rot0-quat'),
                         dtype=float)
            q /= np.linalg.norm(q)
            return q
        return np.array([1.0, 0.0, 0.0, 0.0])

    def should_advance(self, intg):
        if self.dt_ode is None:
            return True
        return intg.tcurr - self.tode_last >= self.dt_ode - self.plugin.tol

    def advance(self, intg, force, moment):
        dt = intg.tcurr - self.tode_last if self.dt_ode else intg.dt

        faccel_new = force / self.mass
        faccel_new *= self._trans_mask

        falpha_new = np.zeros(3)
        if self.ndims == 2:
            falpha_new[2] = moment[0] / self.inertia
        else:
            gyro = np.cross(self.fomega, self.inertia @ self.fomega)
            falpha_new = np.linalg.solve(self.inertia, moment - gyro)
        falpha_new *= self._rot_mask

        # Heun's method
        fomega_new = self.fomega + 0.5*dt*(self.falpha + falpha_new)
        fvelo_new = self.fvelo + 0.5*dt*(self.faccel + faccel_new)

        omega_avg = 0.5*(self.fomega + fomega_new)
        dqdt = 0.5*_quat_mult(self.fquat, np.array([0, *omega_avg]))
        self.fquat = self.fquat + dt*dqdt
        self.fquat /= np.linalg.norm(self.fquat)

        self.floc = self.floc + 0.5*dt*(self.fvelo + fvelo_new)

        self.fomega = fomega_new
        self.falpha = falpha_new
        self.fvelo = fvelo_new
        self.faccel = faccel_new

        self.tode_last = intg.tcurr


class NIRFPlugin(BaseSolverPlugin):
    name = 'nirf'
    systems = 'euler|navier-stokes'
    formulations = 'dual|std'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self._intg = intg

        mode = self.cfg.get(cfgsect, 'motion')
        modecls = subclass_where(BaseMotion, name=mode)
        self.motion = modecls(self, cfgsect)

        if self.motion.needs_force and not self.cfg.hasopt(cfgsect, 'boundary'):
            raise ValueError(
                f"Motion mode '{self.motion.name}' requires 'boundary'"
            )

        self._init_nirf_R()
        if self.motion.extern_names:
            self._register_externs(intg, self.motion.extern_names)

        self._init_force_output(cfgsect)

        if not intg.isrestart:
            self._transform_ics(intg, cfgsect)

        macro = f'nirf_source_{self.ndims}d'
        for eles in intg.system.ele_map.values():
            eles.add_src_macro('pyfr.plugins.solver.kernels.nirf', macro,
                               self.motion.tplargs, ploc=True, soln=True)

    # Read-only proxies so existing self._fX accesses keep working
    @property
    def _floc(self):    return self.motion.floc
    @property
    def _fvelo(self):   return self.motion.fvelo
    @property
    def _faccel(self):  return self.motion.faccel
    @property
    def _fquat(self):   return self.motion.fquat
    @property
    def _fomega(self):  return self.motion.fomega
    @property
    def _falpha(self):  return self.motion.falpha
    @property
    def tode_last(self): return self.motion.tode_last

    def _transform_ics(self, intg, cfgsect):
        ndims = self.ndims
        cor = np.array(self.cfg.getliteral(
            cfgsect, 'center-of-rot', (0.,) * ndims
        ))

        R3 = _quat_to_rotmat(self._fquat)
        R = R3[:ndims, :ndims]
        Rt3 = R3.T
        omega = self._fomega
        velo = self._fvelo[:ndims]
        floc = self._floc[:ndims]

        consts = self.cfg.items_as('constants', float)

        for eles, eb in zip(intg.system.ele_map.values(),
                            intg.system.ele_banks):
            ploc_body = eles.ploc_at_np('upts')
            nupts, neles = ploc_body.shape[0], ploc_body.shape[-1]

            # Inertial position: x_lab = R @ x_body + floc
            ploc_lab = (np.einsum('ij,njk->nik', R, ploc_body)
                        + floc[None, :, None])

            # Re-evaluate IC primitives with x,y,z bound to inertial coords
            vars = dict(consts)
            for i, c in enumerate('xyz'[:ndims]):
                vars[c] = ploc_lab[:, i, :]

            pris = [npeval(self.cfg.getexpr('soln-ics', dv), vars)
                    for dv in eles.privars]
            pris = [np.broadcast_to(v, (nupts, neles)).copy() for v in pris]

            # Velocity transform: u_body = R^T*(u_lab - V_frame) - Ω×r_body
            u_lab = np.stack(pris[1:1 + ndims], axis=1)

            r3 = np.zeros((nupts, 3, neles))
            r3[:, :ndims, :] = ploc_body - cor[None, :, None]
            u3 = np.zeros_like(r3)
            u3[:, :ndims, :] = u_lab - velo[None, :, None]

            ub = np.einsum('ij,njk->nik', Rt3, u3)
            ub -= np.cross(omega, r3.transpose(0, 2, 1)).transpose(0, 2, 1)

            for i in range(ndims):
                pris[1 + i] = ub[:, i, :]

            # Convert primitives to conservatives and push to backend
            s = np.empty((nupts, eles.nvars, neles))
            for i, v in enumerate(eles.pri_to_con(pris, self.cfg)):
                s[:, i, :] = v

            eb[0].set(s)

    def _csv_columns(self):
        comps = 'xyz'[:self.ndims]
        if self.ndims == 2:
            ang = ['phi', 'omega', 'omega_dot']
            moments = ['mz']
        else:
            ang = (['phi', 'theta', 'psi'] +
                   [f'omega_{c}' for c in 'xyz'] +
                   [f'omega_dot_{c}' for c in 'xyz'])
            moments = [f'm{c}' for c in 'xyz']

        return (['t'] + ang +
                [f'loc_{c}' for c in comps] +
                [f'velo_{c}' for c in comps] +
                [f'accel_{c}' for c in comps] +
                [f'f{c}' for c in comps] +
                moments)

    def _csv_header(self):
        return ','.join(self._csv_columns())

    def _init_force_integrator(self, cfgsect):
        intg = self._intg
        comm, rank, root = get_comm_rank_root()

        bcname = self.cfg.get(cfgsect, 'boundary')
        viscous = 'navier-stokes' in intg.system.name

        self.cfg.set(cfgsect, 'quad-deg',
                     self.cfg.getint(cfgsect, 'quad-deg',
                                     self.cfg.getint('solver', 'order')))

        fx0 = np.array(self.cfg.getliteral(cfgsect, 'center-of-rot',
                                           (0.,) * self.ndims), dtype=float)

        bcranks = comm.gather(bcname in intg.system.mesh.bcon, root=root)
        if rank == root and not any(bcranks):
            raise RuntimeError(f'Boundary {bcname} does not exist')

        self._ff_int = NIRFForceIntegrator(
            self.cfg, cfgsect, intg.system, bcname, fx0, viscous)

    def _init_force_output(self, cfgsect):
        if not self.cfg.hasopt(cfgsect, 'boundary'):
            self._ff_int = None
            self._csv = None
            return

        self._init_force_integrator(cfgsect)
        self._ode_nout = self.cfg.getint(cfgsect, 'ode-nout', 1)
        self._ode_count = 0

        _, rank, root = get_comm_rank_root()
        if rank == root and self.cfg.hasopt(cfgsect, 'file'):
            self._csv = init_csv(self.cfg, cfgsect, self._csv_header())
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

    def __call__(self, intg):
        if not self.motion.should_advance(intg):
            return

        need_force = self.motion.needs_force or self._ff_int is not None
        if need_force:
            force, moment = self._ff_int.compute(intg.soln)
        else:
            force, moment = None, None

        self.motion.advance(intg, force, moment)

        if self.motion.has_externs:
            self._update_extern_values()
            self.bind_externs()

        self._update_nirf_R()

        if self._ff_int is not None:
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
            ang = [phi, self._fomega[2], self._falpha[2]]
        else:
            ang = [phi, theta, psi, *self._fomega, *self._falpha]

        self._csv(intg.tcurr, *ang,
                  *self._floc, *self._fvelo, *self._faccel,
                  *force, *moment)

    _sdata_dtype = np.dtype([
        ('loc', 'f8', 3), ('velo', 'f8', 3), ('accel', 'f8', 3),
        ('quat', 'f8', 4), ('omega', 'f8', 3), ('alpha', 'f8', 3),
        ('tode_last', 'f8')
    ])

    def setup(self, sdata, prevcfg, serialiser):
        if sdata is not None:
            ndims = self.ndims
            m = self.motion
            m.floc = np.array(sdata['loc'])[:ndims].copy()
            m.fvelo = np.array(sdata['velo'])[:ndims].copy()
            m.faccel = np.array(sdata['accel'])[:ndims].copy()
            m.fquat = np.array(sdata['quat'])
            m.fomega = np.array(sdata['omega'])
            m.falpha = np.array(sdata['alpha'])
            m.tode_last = float(sdata['tode_last'])
            if m.has_externs:
                self._update_extern_values()

            # Reflect the restored orientation in the backend R matrix
            self._update_nirf_R()

        serialiser.register(self.sprefix,
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
