from collections import defaultdict, deque, namedtuple
import itertools as it
import re
import sys
import time

import numpy as np

from pyfr.cache import memoize
from pyfr.integrators.registers import (RegisterMeta, ScalarRegister,
                                        VectorRegister)
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
from pyfr.plugins import get_plugin
from pyfr.plugins.triggers import TriggerManager
from pyfr.writers.serialise import Serialiser


StepInfo = namedtuple('StepInfo', 'dt action err wtime stages',
                      defaults=(None,))


def _common_plugin_prop(attr):
    def wrapfn(fn):
        @property
        def newfn(self):
            if not (p := getattr(self, attr)):
                wt = self._plugin_wtimes

                t = time.perf_counter()
                p = fn(self)
                wt['common', None] += time.perf_counter() - t

                setattr(self, attr, p)

            return p
        return newfn
    return wrapfn


def kernel_getter(fn):
    @memoize
    def newfn(self, *args, **kwargs):
        return [fn(self, emats, *args, **kwargs)
                for emats in self.system.ele_banks]
    return newfn


class BaseIntegrator(metaclass=RegisterMeta):
    def __init__(self, backend, mesh, initsoln, cfg):
        self.backend = backend
        self.isrestart = initsoln is not None
        self.cfg = cfg
        prevcfgs = initsoln.prevcfgs if initsoln else {}
        self.prevcfgs = {k: v.tostr() for k, v in prevcfgs.items()}

        # Start time
        self.tstart = cfg.getfloat('solver-time-integrator', 'tstart', 0.0)
        self.tend = cfg.getfloat('solver-time-integrator', 'tend')

        # Current time; defaults to tstart unless restarting
        if self.isrestart:
            stats = initsoln.stats
            self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
        else:
            self.tcurr = self.tstart

        # Kahan compensated summation for tcurr accumulation
        self._tcurr_comp = 0.0

        # List of target times to advance to
        self.tlist = deque([self.tend])

        # Accepted and rejected step counters
        self.nacptsteps = 0
        self.nrjctsteps = 0
        self.nrhsevals = 0

        # Current and minimum time steps
        self.dt = cfg.getfloat('solver-time-integrator', 'dt')
        self.dtmin = cfg.getfloat('solver-time-integrator', 'dt-min', 1e-12)

        # Look-ahead window for smoothing dt near target times
        self._dt_lookhead = cfg.getint('solver-time-integrator',
                                       'dt-lookahead', 10)

        # Extract the UUID of the mesh (to be saved with solutions)
        self.mesh_uuid = mesh.uuid

        self._invalidate_caches()

        # Record the starting wall clock time
        self._wstart = time.perf_counter()

        # Record the total amount of time spent in each plugin
        self._plugin_wtimes = defaultdict(lambda: 0)

        # Abort computation
        self._abort = False
        self._abort_reason = ''

        # Saving serialised data to checkpoint files
        self.serialiser = Serialiser()

        # Trigger manager for conditional plugin execution
        self.triggers = TriggerManager()

    def _rhs(self, t, uin, uout):
        self.system.rhs(t, uin, uout)
        self.nrhsevals += 1

    def plugin_abort(self, reason):
        self._abort = True
        self._abort_reason = self._abort_reason or reason

    def plugin_end(self):
        self.tend = self.tcurr

    def fire_trigger(self, name):
        self.triggers.fire(name)

    def _get_plugins(self, initsoln):
        # Parse trigger sections (system exists, ele_map available)
        self.triggers.parse_config(self)

        # Restore trigger state from checkpoint if restarting
        if initsoln:
            self.triggers.restore(initsoln.state)

        plugins = []
        prevcfg = initsoln.config if initsoln else None

        for s in self.cfg.sections():
            if (m := re.match('(soln|solver)-plugin-(.+?)(?:-(.+))?$', s)):
                cfgsect, ptype, name, suffix = m[0], m[1], m[2], m[3]

                if ptype == 'solver' and suffix:
                    raise ValueError(f'solver-plugin-{name} cannot have a '
                                     'suffix')

                args = (ptype, name, self, cfgsect)
                if ptype == 'soln':
                    args += (suffix, )

                # Instantiate
                plugin = get_plugin(*args)
                sprefix = plugin.sprefix
                sdata = initsoln.state.get(sprefix) if initsoln else None
                plugin.setup(sdata, prevcfg, self.serialiser)
                plugins.append(plugin)

        # Validate plugin trigger references
        refs = []
        for p in plugins:
            if p.trigger:
                refs.extend(p.trigger)
            if p.trigger_write_name:
                refs.append(p.trigger_write_name)
            if p.trigger_fire_name:
                refs.append(p.trigger_fire_name)

        self.triggers.check_names(refs)

        # Restore and register plugin trigger activation states
        self._init_plugin_activated(plugins, initsoln)

        return plugins

    def _init_plugin_activated(self, plugins, initsoln):
        # Identify plugins using activate mode
        aplugins = [p for p in plugins
                    if p.trigger and p.trigger_action == 'activate']

        if not aplugins:
            return

        # Restore activation states from checkpoint
        if initsoln and (sd := initsoln.state.get('plugins/activated')) is not None:
            lookup = {r['name'].decode(): r['active'] for r in sd}
            for p in aplugins:
                if (v := lookup.get(p.sprefix)) is not None:
                    p.trigger_activated = v

        # Register serialisation
        _, rank, root = get_comm_rank_root()

        def datafn():
            data = [(p.sprefix, p.trigger_activated) for p in aplugins]
            n = max(len(p.sprefix) for p in aplugins)
            return np.array(data, dtype=[('name', f'S{n}'), ('active', '?')])

        self.serialiser.register('plugins/activated',
                                 datafn if rank == root else None)

    def _run_plugins(self):
        wtimes = self._plugin_wtimes

        self.backend.wait()

        # Evaluate trigger conditions
        if self.triggers:
            self.triggers.evaluate(self)

        # Fire off the plugins and tally up the runtime
        for plugin in self.plugins:
            # Skip disabled plugins
            if not plugin.enabled:
                continue

            tstart = time.perf_counter()
            tcommon = wtimes['common', None]

            # Handle trigger-write independently of normal scheduling
            if (tw := plugin.trigger_write_name) and self.triggers.active(tw):
                plugin.trigger_write(self)

            # Normal scheduling
            if self._plugin_should_run(plugin):
                plugin(self)

                if ts := plugin.trigger_fire_name:
                    self.triggers.fire(ts)

            tend = time.perf_counter()
            dt = tend - tstart - wtimes['common', None] + tcommon

            wtimes[plugin.name, plugin.suffix] += dt

        # Abort if plugins request it
        self._check_abort()

    def _plugin_should_run(self, plugin):
        # Trigger scheduling
        if plugin.trigger is not None:
            active = plugin.trigger_comb(self.triggers)

            # Gate mode
            if plugin.trigger_action == 'gate':
                return active

            # Activate mode; latch on first activation
            if active:
                plugin.trigger_activated = True

            if not plugin.trigger_activated:
                return False

        # Steps/dt-out scheduling
        return not plugin.nsteps or self.nacptsteps % plugin.nsteps == 0

    def _finalise_plugins(self):
        for plugin in self.plugins:
            plugin.finalise(self)

    def call_plugin_dt(self, tstart, dt):
        ta = self.tlist
        tbegin = tstart if tstart > self.tcurr else self.tcurr
        tb = deque(np.arange(tbegin, self.tend, dt).tolist())

        self.tlist = tlist = deque()

        # Merge the current and new time lists
        while ta and tb:
            t = ta.popleft() if ta[0] < tb[0] else tb.popleft()
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

        for t in it.chain(ta, tb):
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

    def _invalidate_caches(self):
        self._curr_soln = None
        self._curr_grad_soln = None
        self._curr_dt_soln = None
        self._grads_current = False

    def compute_grads(self):
        if not self._grads_current:
            self.system.compute_grads(self.tcurr, self.idxcurr)
            self._grads_current = True

    def _advance_time(self, dt):
        # Kahan compensated summation to avoid accumulation error
        y = dt - self._tcurr_comp
        t = self.tcurr + y
        self._tcurr_comp = (t - self.tcurr) - y
        self.tcurr = t

    def _clamp_dt(self, dt_want, t):
        remaining = t - self.tcurr
        nsteps = -(-remaining // dt_want)

        if nsteps > self._dt_lookhead:
            return dt_want
        else:
            return max(remaining / nsteps, self.dtmin)

    def step(self, t, dt):
        pass

    def _timed_step(self, t, dt):
        comm, _, _ = get_comm_rank_root()
        tstart = time.perf_counter()

        with self.backend.region('step'):
            idxs = self.step(t, dt)
            self.backend.wait()

        # Compute the wall time
        wtime = time.perf_counter() - tstart
        wtime = scal_coll(comm.Allreduce, wtime, op=mpi.MAX)

        return idxs, wtime

    def advance_to(self, t):
        pass

    def run(self):
        for t in self.tlist:
            self.advance_to(t)

        self._finalise_plugins()

    @property
    def nsteps(self):
        return self.nacptsteps + self.nrjctsteps

    def collect_stats(self, stats):
        wtime = time.perf_counter() - self._wstart

        # Simulation and wall clock times
        stats.set('solver-time-integrator', 'tcurr', self.tcurr)
        stats.set('solver-time-integrator', 'wall-time', wtime)

        # Plugin wall clock times
        for (pname, psuffix), t in self._plugin_wtimes.items():
            k = f'plugin-wall-time-{pname}'
            if psuffix:
                k += f'-{psuffix}'

            stats.set('solver-time-integrator', k, t)

        # Step and function evaluation counts
        stats.set('solver-time-integrator', 'nsteps', self.nsteps)
        stats.set('solver-time-integrator', 'nacptsteps', self.nacptsteps)
        stats.set('solver-time-integrator', 'nrjctsteps', self.nrjctsteps)
        stats.set('solver-time-integrator', 'nrhsevals', self.nrhsevals)
        stats.set('solver-time-integrator', 'rhs-gdof/s',
                  self.gndofs * self.nrhsevals / wtime)

        # MPI wait times
        if self.cfg.getbool('backend', 'collect-wait-times', False):
            comm, _, _ = get_comm_rank_root()

            wait_times = comm.allgather(self.system.rhs_wait_times())
            for i, ms in enumerate(zip(*wait_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-wait-times', f'rhs-graph-{i}-{k}',
                              ','.join(f'{v[j]:.3g}' for v in ms))

        # Backend memory
        comm, _, _ = get_comm_rank_root()
        mem_info = comm.allgather(self.backend.memory_info())
        stats.set('backend-memory', 'current',
                  ','.join(str(m.current) for m in mem_info))
        stats.set('backend-memory', 'peak',
                  ','.join(str(m.peak) for m in mem_info))

    @property
    def cfgmeta(self):
        cfg = self.cfg.tostr()

        # Build config chain
        n = len(self.prevcfgs)
        ret = dict(self.prevcfgs, config=cfg)

        # Add current config to chain if it differs from the previous one
        if n == 0 or cfg != ret[f'config-{n - 1}']:
            ret[f'config-{n}'] = cfg

        return ret

    def _check_abort(self):
        comm, _, _ = get_comm_rank_root()

        if scal_coll(comm.Allreduce, int(self._abort), op=mpi.LOR):
            self._finalise_plugins()

            reason = self._abort_reason
            sys.exit(comm.allreduce(reason, op=lambda x, y: x or y))

    def _get_gndofs(self):
        comm, _, _ = get_comm_rank_root()

        # Get the number of degrees of freedom in this partition
        ndofs = sum(self.system.ele_ndofs)

        # Sum to get the global number over all partitions
        return scal_coll(comm.Allreduce, ndofs)

    @kernel_getter
    def _get_add_kerns(self, emats, *rs, in_scale=(), in_scale_idxs=(),
                       out_scale=()):
        return self.backend.kernel('axnpby', *[emats[r] for r in rs],
                                   in_scale=in_scale, out_scale=out_scale,
                                   in_scale_idxs=in_scale_idxs)

    def _addv(self, consts, regidxs, in_scale=(), in_scale_idxs=(),
              out_scale=()):
        if len(regidxs) != len(set(regidxs)):
            raise ValueError('Duplicate register indices')

        # Get a suitable set of axnpby kernels
        in_s, out_s = tuple(in_scale), tuple(out_scale)
        axnpby = self._get_add_kerns(*regidxs, in_scale=in_s,
                                     in_scale_idxs=in_scale_idxs,
                                     out_scale=out_s)

        # Bind the arguments
        for k in axnpby:
            k.bind(*consts)

        self.backend.run_kernels(axnpby)

    def _add(self, *args, in_scale=(), in_scale_idxs=(), out_scale=()):
        self._addv(args[::2], args[1::2], in_scale, in_scale_idxs, out_scale)

    def _size_register(self, reg, n):
        if not reg.dynamic:
            raise ValueError('Only dynamic registers can be sized')

        if not reg.vector and n > 1:
            raise ValueError('Dynamic scalar register must be 0 or 1')

        if reg.vector:
            new_reg = VectorRegister(n=n, rhs=reg.rhs, extent=reg.extent)
        else:
            new_reg = ScalarRegister(n=n, rhs=reg.rhs, extent=reg.extent)

        self._registers[new_reg] = self._registers.pop(reg)

    def _assign_registers(self):
        off = 0

        # Allocate register numbers to member variables starting with
        # registers which can be used for RHS evaluations
        for rhs in [True, False]:
            for r, name in self._registers.items():
                if r.rhs == rhs and r.n != 'dyn':
                    if r.vector:
                        v = list(range(off, off + r.n))
                    elif r.n == 1:
                        v = off
                    else:
                        v = None
                    setattr(self, name, v)
                    off += r.n

    @property
    def _nregs(self):
        def nregs(rhs):
            return sum(r.n for r in self._registers
                       if r.rhs == rhs and r.n != 'dyn')

        return nregs(True), nregs(False)
