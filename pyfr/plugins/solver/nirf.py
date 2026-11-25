import math

import numpy as np

from pyfr.dsl import CodeGenerator, Simplifier, parse_expr
from pyfr.dsl.derivs import Differentiator
from pyfr.dsl.nodes import Float, Int, Ternary, Var, walk_expr
from pyfr.dsl.rewriter import rename_vars
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import Quaternion
from pyfr.plugins.mixins import SeriesWriterMixin
from pyfr.plugins.soln.fluidforce import FluidForceIntegrator
from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.util import ndrange, subclass_where


class _HostCodeGenerator(CodeGenerator):
    def _generate_expr(self, expr):
        match expr:
            case Ternary(cond, true, false):
                c, a, b = map(self._generate_expr, (cond, true, false))

                return f'({a} if {c} else {b})'
            case _:
                return super()._generate_expr(expr)


def _subst(tpl, subs):
    return rename_vars(parse_expr(tpl), subs)


class BaseMotion:
    name = None
    needs_force = False

    def __init__(self, intg, cfgsect, tol):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect
        self.ndims = intg.system.ndims
        self.tol = tol

    @property
    def rmat(self):
        return self.fquat.to_rotmat()

    def externs(self):
        return {}

    def schedule(self, intg):
        pass

    def should_advance(self, t):
        return True

    def advance(self, t, force=None, moment=None):
        raise NotImplementedError


class PrescribedMotion(BaseMotion):
    name = 'prescribed'

    def __init__(self, intg, cfgsect, tol):
        super().__init__(intg, cfgsect, tol)

        self._simp = Simplifier().simplify_fully

        # Differentiate with respect to t alone
        self._diff = Differentiator({'t'}, set(), set(), lambda n: f'_d_{n}')

        self.kin = self._kinematics()

        # Compile the expressions for evaluation on the host
        gen = _HostCodeGenerator().generate
        self._hexprs = {k: compile(gen(v), k, 'eval')
                        for k, v in self.kin.items()}

        self._hsyms = {**vars(math), 'fmin': min, 'fmax': max,
                       '__builtins__': {}}

        self.advance(intg.tcurr)

    def _cfgexpr(self, name):
        cfg, cfgsect = self.cfg, self.cfgsect

        csubs = cfg.items('constants') | {'pi': str(math.pi)}
        expr = parse_expr(cfg.getexpr(cfgsect, name, '0.0', subs=csubs))

        # Ensure the expression depends upon t alone
        for n in walk_expr(expr):
            if isinstance(n, Var) and n.name != 't':
                raise ValueError(f'Unknown variable {n.name!r} in option '
                                 f'{name!r} of [{cfgsect}]')

        return self._simp(expr)

    def _ddt(self, expr):
        dexpr = self._diff.diff(expr)

        # Seed the tangent of t with one
        return self._simp(rename_vars(dexpr, {'_d_t': Int(1)}))

    def _translation(self):
        kin = {}
        for c in 'xyz'[:self.ndims]:
            kin[f'frame_loc_{c}'] = loc = self._cfgexpr(f'frame-loc-{c}')
            kin[f'frame_velo_{c}'] = velo = self._ddt(loc)
            kin[f'frame_accel_{c}'] = self._ddt(velo)

        return kin

    def _rotation(self):
        ndims, simp, d = self.ndims, self._simp, self._ddt

        if ndims == 2:
            rot = {'rx': Float(0.0), 'ry': Float(0.0),
                   'rz': self._cfgexpr('frame-rot-z')}
            omega = ['0.0', '0.0', 'drz']
            rmat = [['cz', '-sz'], ['sz', 'cz']]
        else:
            rot = {f'r{c}': self._cfgexpr(f'frame-rot-{c}') for c in 'xyz'}

            # Express the body-frame angular velocity by the ZYX Euler rates
            omega = ['drx - drz*sy', 'dry*cx + drz*sx*cy',
                     'drz*cx*cy - dry*sx']

            # Express the rotation matrix by the ZYX Euler angles
            rmat = [['cz*cy', 'cz*sy*sx - sz*cx', 'cz*sy*cx + sz*sx'],
                    ['sz*cy', 'sz*sy*sx + cz*cx', 'sz*sy*cx - cz*sx'],
                    ['-sy', 'cy*sx', 'cy*cx']]

        # Substitute the sines, cosines and rates of the angles
        subs = {f'{f[0]}{c}': _subst(f'{f}(r{c})', rot)
                for f in ('sin', 'cos') for c in 'xyz'}
        subs |= {f'd{k}': d(v) for k, v in rot.items()}

        kin = {}
        for c, w in zip('xyz', omega):
            w = simp(_subst(w, subs))

            kin[f'frame_rot_{c}'] = rot[f'r{c}']
            kin[f'frame_omega_{c}'] = w
            kin[f'frame_alpha_{c}'] = d(w)

        for i, j in ndrange(ndims, ndims):
            kin[f'frame_rmat_{i}{j}'] = simp(_subst(rmat[i][j], subs))

        return kin

    def _body_accel(self, kin):
        ndims = self.ndims
        comps = 'xyz'[:ndims]

        # Resolve the linear acceleration in body axes, R^T*a
        subs = {f'a{j}': kin[f'frame_accel_{c}'] for j, c in enumerate(comps)}
        subs |= {f'R{i}{j}': kin[f'frame_rmat_{i}{j}']
                 for i, j in ndrange(ndims, ndims)}

        bkin = {}
        for i, c in enumerate(comps):
            ba = ' + '.join(f'R{j}{i}*a{j}' for j in range(ndims))
            bkin[f'frame_baccel_{c}'] = self._simp(_subst(ba, subs))

        return bkin

    def _kinematics(self):
        kin = self._translation() | self._rotation()

        return kin | self._body_accel(kin)

    def advance(self, t, force=None, moment=None):
        comps = 'xyz'[:self.ndims]

        def state(kind, comps):
            exprs = [self._hexprs[f'frame_{kind}_{c}'] for c in comps]

            return np.array([eval(e, self._hsyms, {'t': t}) for e in exprs],
                            dtype=float)

        self.floc, self.fvelo = state('loc', comps), state('velo', comps)
        self.faccel = state('accel', comps)
        self.fquat = Quaternion.from_euler(*state('rot', 'xyz'))
        self.fomega, self.falpha = state('omega', 'xyz'), state('alpha', 'xyz')

        self.tode_last = t


class FreeMotion(BaseMotion):
    name = 'free'
    needs_force = True

    def __init__(self, intg, cfgsect, tol):
        super().__init__(intg, cfgsect, tol)

        cfg, ndims = self.cfg, self.ndims

        self._parse_dof()

        self.mass = cfg.getfloat(cfgsect, 'mass')
        if ndims == 2:
            self.inertia = cfg.getfloat(cfgsect, 'inertia')
        else:
            inertia = cfg.getliteral(cfgsect, 'inertia')
            self.inertia = np.array(inertia).reshape(3, 3)

        def vec(name):
            v = cfg.getliteral(cfgsect, name, (0,)*ndims)

            return np.array(v, dtype=float)

        self.floc, self.fvelo = vec('frame-loc0'), vec('frame-velo0')
        self.fquat = self._parse_rot0()
        self.fomega = self._rvec('frame-omega0')

        # Leave the accelerations for the first update to take from the loads
        self.faccel, self.falpha = np.zeros(ndims), np.zeros(3)

        if cfg.hasopt(cfgsect, 'dt-ode'):
            self.dt_ode = cfg.getfloat(cfgsect, 'dt-ode')
        else:
            self.dt_ode = None

        self.tode_last = intg.tcurr

        # Refer to the frame parameters as runtime kernel arguments
        self.kin = {k: Var(k) for k in self.externs()}
        self.kin |= {f'frame_rmat_{i}{j}': parse_expr(f'nirf_R[{i}][{j}]')
                     for i, j in ndrange(ndims, ndims)}

        # Zero the rates about x and y in 2D
        if ndims == 2:
            self.kin |= {f'frame_{k}_{c}': Float(0.0)
                         for k in ('omega', 'alpha') for c in 'xy'}

    def _rvec(self, name):
        cfg, cfgsect = self.cfg, self.cfgsect

        if self.ndims == 3:
            v = cfg.getliteral(cfgsect, name, (0,)*3)
        elif np.isscalar(v := cfg.getliteral(cfgsect, name, 0)):
            v = [0, 0, v]
        else:
            raise ValueError(f'{name} must be a scalar in 2D')

        return np.array(v, dtype=float)

    def _parse_dof(self):
        comps = 'xyz'[:self.ndims]
        rcomps = 'z' if self.ndims == 2 else 'xyz'
        all_dof = [*comps, *(f'r{c}' for c in rcomps)]

        # Free every degree of freedom unless told otherwise
        dof_str = self.cfg.get(self.cfgsect, 'dof', ','.join(all_dof))
        if dof_str.strip().lower() in ('', 'none'):
            free_dof = set()
        else:
            free_dof = {s.strip() for s in dof_str.split(',')}
            if invalid := free_dof - set(all_dof):
                raise ValueError(f'Invalid DOF: {invalid}. Valid: {all_dof}')

        self._trans_mask = np.array([c in free_dof for c in comps])
        self._rot_mask = np.array([f'r{c}' in free_dof for c in 'xyz'])

    def _parse_rot0(self):
        cfg, cfgsect = self.cfg, self.cfgsect

        if cfg.hasopt(cfgsect, 'frame-rot0-euler'):
            return Quaternion.from_euler(*self._rvec('frame-rot0-euler'))
        elif cfg.hasopt(cfgsect, 'frame-rot0-quat'):
            q = Quaternion(cfg.getliteral(cfgsect, 'frame-rot0-quat'))

            if self.ndims == 2 and q.q[1:3].any():
                raise ValueError('frame-rot0-quat must be about z in 2D')

            return q.normalised()
        else:
            return Quaternion([1, 0, 0, 0])

    def externs(self):
        ndims = self.ndims

        # Resolve the linear acceleration in body axes for the momentum source
        baccel = self.rmat[:ndims, :ndims].T @ self.faccel

        ev = {}
        for i, c in enumerate('xyz'[:ndims]):
            ev[f'frame_loc_{c}'] = self.floc[i]
            ev[f'frame_velo_{c}'] = self.fvelo[i]
            ev[f'frame_baccel_{c}'] = baccel[i]

        # Pass only the rates about z in 2D
        for i, c in enumerate('xyz'):
            if ndims == 3 or c == 'z':
                ev[f'frame_omega_{c}'] = self.fomega[i]
                ev[f'frame_alpha_{c}'] = self.falpha[i]

        return ev

    def schedule(self, intg):
        # Ask to be called dt-ode after the last update
        if self.dt_ode is not None:
            intg.call_plugin_dt(self.tode_last + self.dt_ode, self.dt_ode)

    def should_advance(self, t):
        dt = t - self.tode_last

        # Advance over a zero interval too, which sets the first accelerations
        return (self.dt_ode is None or dt == 0 or
                dt >= self.dt_ode - self.tol)

    def advance(self, t, force, moment):
        ndims = self.ndims

        dt = t - self.tode_last

        if ndims == 2:
            falpha = np.array([0, 0, moment[0] / self.inertia])
            falpha *= self._rot_mask
        else:
            gyro = np.cross(self.fomega, self.inertia @ self.fomega)

            # Solve for the angular acceleration about the free axes alone
            m = self._rot_mask
            falpha = np.zeros(3)
            falpha[m] = np.linalg.solve(self.inertia[np.ix_(m, m)],
                                        (moment - gyro)[m])

        # Advance the angular velocity and orientation with Heun's method
        fomega = self.fomega + 0.5*dt*(self.falpha + falpha)

        omega_avg = 0.5*(self.fomega + fomega)
        dqdt = 0.5*(self.fquat*Quaternion([0, *omega_avg])).q
        self.fquat = Quaternion(self.fquat.q + dt*dqdt).normalised()

        # Rotate the body-axis force into lab axes for the acceleration
        faccel = self.rmat[:ndims, :ndims] @ force / self.mass
        faccel *= self._trans_mask

        fvelo = self.fvelo + 0.5*dt*(self.faccel + faccel)
        self.floc = self.floc + 0.5*dt*(self.fvelo + fvelo)

        self.fomega, self.falpha = fomega, falpha
        self.fvelo, self.faccel = fvelo, faccel

        self.tode_last = t


class NIRFPlugin(SeriesWriterMixin, BaseSolverPlugin):
    name = 'nirf'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'

    _sdata_dtype = np.dtype([
        ('loc', 'f8', 3), ('velo', 'f8', 3), ('accel', 'f8', 3),
        ('quat', 'f8', 4), ('omega', 'f8', 3), ('alpha', 'f8', 3),
        ('tode_last', 'f8')
    ])

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        cfg, ndims = self.cfg, self.ndims

        # Track the time the frame was last advanced to
        self._tadv = None

        # Cache the loads with the time they were integrated at
        self._loads = (None, None)

        # Read the centre of rotation, about which moments are also taken
        origin = cfg.getliteral(cfgsect, 'center-of-rot', (0,)*ndims)
        self.origin = np.array(origin, dtype=float)

        motioncls = subclass_where(BaseMotion, name=cfg.get(cfgsect, 'motion'))
        self.motion = m = motioncls(intg, cfgsect, self.tol)

        if (sdata := intg.initstate.get(self.sprefix)) is not None:
            self._restore_state(sdata)

        m.schedule(intg)

        intg.serialiser.register(self.sprefix,
                                 lambda: self._serialise_data(intg))

        self._transform_bcs(intg)

        # Bind any runtime frame parameters to the kernels
        if (names := list(m.externs())):
            self._register_externs(intg, names)

        self._init_loads(intg)

        if not intg.isrestart:
            self._transform_ics(intg)

        gen = CodeGenerator().generate

        def texts(kind, comps):
            return [gen(m.kin[f'frame_{kind}_{c}']) for c in comps]

        # Render the frame parameters for the source term kernels
        tplargs = {
            'frame_omega': texts('omega', 'xyz'),
            'frame_alpha': texts('alpha', 'xyz'),
            'frame_baccel': texts('baccel', 'xyz'[:ndims]),
            'frame_origin': self.origin.tolist()
        }

        for eles in intg.system.ele_map.values():
            eles.add_src_macro('pyfr.plugins.solver.kernels.macros.nirf',
                               'nirf_source', tplargs, ploc=True, soln=True)

    def _restore_state(self, sdata):
        ndims, m = self.ndims, self.motion

        m.floc = np.array(sdata['loc'][:ndims])
        m.fvelo = np.array(sdata['velo'][:ndims])
        m.faccel = np.array(sdata['accel'][:ndims])
        m.fquat = Quaternion(sdata['quat'])
        m.fomega = np.array(sdata['omega'])
        m.falpha = np.array(sdata['alpha'])
        m.tode_last = float(sdata['tode_last'])

    def _body_frame(self, kin, pos):
        ndims = self.ndims
        comps, vkeys = 'xyz'[:ndims], 'uvw'[:ndims]
        simp, gen = Simplifier().simplify_fully, CodeGenerator().generate

        subs = {f'X{i}': Float(v) for i, v in enumerate(self.origin.tolist())}
        subs |= {f'P{i}': p for i, p in enumerate(pos)}
        subs |= {f'L{i}': kin[f'frame_loc_{c}'] for i, c in enumerate(comps)}
        subs |= {f'V{i}': kin[f'frame_velo_{c}'] for i, c in enumerate(comps)}
        subs |= {f'W{i}': kin[f'frame_omega_{c}'] for i, c in enumerate('xyz')}
        subs |= {f'R{i}{j}': kin[f'frame_rmat_{i}{j}']
                 for i, j in ndrange(ndims, ndims)}

        # Express the lab-frame position, x_lab = R*(x_body - x0) + x0 + loc
        xlab = []
        for i in range(ndims):
            rx = ' + '.join(f'R{i}{j}*(P{j} - X{j})' for j in range(ndims))
            xlab.append(_subst(f'{rx} + X{i} + L{i}', subs))

        def lab_position(node):
            if node in pos:
                return xlab[pos.index(node)]
            else:
                return node.map(lab_position)

        # Form r x omega, with r zero out of plane in 2D
        r = [f'(P{i} - X{i})' if i < ndims else '0.0' for i in range(3)]
        rxw = [f'W{k}*{r[j]} - W{j}*{r[k]}'
               for j, k in [(1, 2), (2, 0), (0, 1)][:ndims]]

        # Express the body-frame velocity, u_body = R^T*(u_lab - V) - omega x r
        ubody = []
        for i, c in enumerate(rxw):
            rtu = ' + '.join(f'R{j}{i}*(U{j} - V{j})' for j in range(ndims))
            ubody.append(f'{rtu} + {c}')

        def body_frame(cfgsect, exprs):
            exprs = {k: parse_expr(v) for k, v in exprs.items()}

            # Evaluate position-dependent expressions at lab-frame points
            lab = {k: lab_position(e) for k, e in exprs.items()}
            rexprs = {k: e for k, e in lab.items() if e is not exprs[k]}

            # Take a velocity given with other variables as a lab-frame one
            if set(vkeys) < exprs.keys():
                usubs = subs | {f'U{i}': lab[k] for i, k in enumerate(vkeys)}
                rexprs |= {k: _subst(u, usubs) for k, u in zip(vkeys, ubody)}

            # Parenthesise, as these are pasted into larger expressions
            return {k: f'({gen(simp(e))})' for k, e in rexprs.items()}

        return body_frame

    def _transform_bcs(self, intg):
        pos = [parse_expr(f'ploc[{i}]') for i in range(self.ndims)]
        body_frame = self._body_frame(self.motion.kin, pos)

        # Offer the frame parameters as externs to the rewritten expressions
        externs = {k: ('scalar fpdtype_t', None)
                   for k in self.motion.externs()}
        if externs:
            self._nirf_R = intg.backend.matrix((3, 3),
                                               initval=self.motion.rmat)
            externs['nirf_R'] = ('in broadcast fpdtype_t[3][3]', self._nirf_R)

        intg.system.rewrite_bc_exprs(body_frame, externs)

    def _update_extern_values(self):
        self._extern_values = self.motion.externs()
        self._nirf_R.set(self.motion.rmat)

    def _init_loads(self, intg):
        cfg, cfgsect = self.cfg, self.cfgsect

        # Write output only if given a file and a boundary for the loads
        self._has_output = (cfg.hasopt(cfgsect, 'file') and
                            cfg.hasopt(cfgsect, 'boundary'))

        # Set up the load integration if the body or the output needs it
        if self.motion.needs_force or self._has_output:
            if not cfg.hasopt(cfgsect, 'quad-deg'):
                order = cfg.getint('solver', 'order')
                cfg.set(cfgsect, 'quad-deg', max(order, 1))

            self._ff_int = FluidForceIntegrator(
                intg, cfgsect, cfg.get(cfgsect, 'boundary'), self.origin
            )

        if self._has_output:
            if cfg.hasopt(cfgsect, 'dt-out'):
                self.dt_out = cfg.getfloat(cfgsect, 'dt-out')
                self.tout_last = intg.tcurr

                # Register our output times with the integrator
                intg.call_plugin_dt(intg.tcurr, self.dt_out)
            else:
                self.dt_out = None
                self.nsteps_out = cfg.getint(cfgsect, 'nsteps-out', 1)

            # The root rank needs to open the output file
            _, rank, root = get_comm_rank_root()
            if rank == root:
                self._init_series(intg, self._fields)

    @property
    def _fields(self):
        comps = 'xyz'[:self.ndims]
        rcomps = 'z' if self.ndims == 2 else 'xyz'

        fields = [f'{k}_{c}' for k in ('rot', 'omega', 'omega_dot')
                  for c in rcomps]
        fields += [f'{k}_{c}' for k in ('loc', 'velo', 'accel') for c in comps]

        return fields + [f'f{c}' for c in comps] + [f'm{c}' for c in rcomps]

    def _transform_ics(self, intg):
        ndims, m = self.ndims, self.motion

        # Freeze the frame parameters at their values at the start of the run
        state = {'loc': m.floc, 'velo': m.fvelo, 'omega': m.fomega}
        kin = {f'frame_{k}_{c}': Float(float(v))
               for k, vals in state.items() for c, v in zip('xyz', vals)}
        kin |= {f'frame_rmat_{i}{j}': Float(float(m.rmat[i, j]))
                for i, j in ndrange(ndims, ndims)}

        body_frame = self._body_frame(kin, [Var(c) for c in 'xyz'[:ndims]])

        intg.system.rewrite_ic_exprs(body_frame)

    def _integrate_loads(self, intg):
        if self._loads[0] != intg.tcurr:
            comm, rank, root = get_comm_rank_root()

            # Sum the pressure and viscous loads over all ranks
            fm = self._ff_int(intg).sum(axis=0)
            comm.Allreduce(mpi.IN_PLACE, fm, op=mpi.SUM)

            self._loads = (intg.tcurr, (fm[:self.ndims], fm[self.ndims:]))

        return self._loads[1]

    def _advance(self, intg):
        t, m = intg.tcurr, self.motion

        if self._tadv != t and m.should_advance(t):
            loads = self._integrate_loads(intg) if m.needs_force else ()
            m.advance(t, *loads)

            if self._extern_values:
                self._update_extern_values()
                self.bind_externs(intg)

            self._tadv = t

    def _output_due(self, intg):
        if not self._has_output:
            return False
        elif self.dt_out is None:
            return intg.nacptsteps % self.nsteps_out == 0
        else:
            return intg.tcurr - self.tout_last >= self.dt_out - self.tol

    def __call__(self, intg):
        self._advance(intg)

        if self._output_due(intg):
            self.tout_last = intg.tcurr

            force, moment = self._integrate_loads(intg)

            _, rank, root = get_comm_rank_root()
            if rank == root:
                m, nr = self.motion, len(moment)
                euler = m.fquat.to_euler()

                # Lay the row out as angular state, linear state, then loads
                row = [*euler[-nr:], *m.fomega[-nr:], *m.falpha[-nr:],
                       *m.floc, *m.fvelo, *m.faccel, *force, *moment]

                self._write(intg.tcurr, row)

    def _serialise_data(self, intg):
        # Bring the frame up to date in case a writer runs first
        self._advance(intg)

        m, pad = self.motion, (0, 3 - self.ndims)

        # Pad the linear vectors to three components for the saved state
        lin = [np.pad(v, pad) for v in (m.floc, m.fvelo, m.faccel)]
        sdata = (*lin, m.fquat.q, m.fomega, m.falpha, m.tode_last)

        return np.void(sdata, dtype=self._sdata_dtype)
