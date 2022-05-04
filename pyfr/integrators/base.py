# -*- coding: utf-8 -*-

from collections import deque
import itertools as it
import re
import sys
import time

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins import get_plugin
from pyfr.util import memoize


class BaseIntegrator:
    def __init__(self, backend, rallocs, mesh, initsoln, cfg):
        self.backend = backend
        self.rallocs = rallocs
        self.isrestart = initsoln is not None
        self.cfg = cfg
        self.prevcfgs = {f: initsoln[f] for f in initsoln or []
                         if f.startswith('config-')}

        # Start time
        self.tstart = cfg.getfloat('solver-time-integrator', 'tstart', 0.0)
        self.tend = cfg.getfloat('solver-time-integrator', 'tend')

        # Current time; defaults to tstart unless restarting
        if self.isrestart:
            stats = Inifile(initsoln['stats'])
            self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
        else:
            self.tcurr = self.tstart

        # List of target times to advance to
        self.tlist = deque([self.tend])

        # Accepted and rejected step counters
        self.nacptsteps = 0
        self.nrjctsteps = 0
        self.nacptchain = 0

        # Current and minimum time steps
        self._dt = cfg.getfloat('solver-time-integrator', 'dt')
        self.dtmin = cfg.getfloat('solver-time-integrator', 'dt-min', 1e-12)

        # Extract the UUID of the mesh (to be saved with solutions)
        self.mesh_uuid = mesh['mesh_uuid']

        # Get a queue for subclasses to use
        self._queue = backend.queue()

        # Solution cache
        self._curr_soln = None

        # Solution gradients cache
        self._curr_grad_soln = None

        # Record the starting wall clock time
        self._wstart = time.time()

        # Abort computation
        self.abort = False

    def _get_plugins(self, initsoln):
        plugins = []

        for s in self.cfg.sections():
            if (m := re.match('soln-plugin-(.+?)(?:-(.+))?$', s)):
                cfgsect, name, suffix = m[0], m[1], m[2]

                data = {}
                if initsoln is not None:
                    # Get the plugin data stored in the solution, if any
                    prefix = self.get_plugin_data_prefix(name, suffix)
                    for f in initsoln:
                        if f.startswith(f'{prefix}/'):
                            data[f.split('/')[2]] = initsoln[f]

                # Instantiate
                plugins.append(get_plugin(name, self, cfgsect, suffix, **data))

        return plugins

    @staticmethod
    def get_plugin_data_prefix(name, suffix):
        if suffix:
            return f'plugins/{name}-{suffix}'
        else:
            return f'plugins/{name}'

    def call_plugin_dt(self, dt):
        ta = self.tlist
        tb = deque(np.arange(self.tcurr, self.tend, dt).tolist())

        self.tlist = tlist = deque()

        # Merge the current and new time lists
        while ta and tb:
            t = ta.popleft() if ta[0] < tb[0] else tb.popleft()
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

        for t in it.chain(ta, tb):
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

    def step(self, t, dt):
        pass

    def advance_to(self, t):
        pass

    def run(self):
        for t in self.tlist:
            self.advance_to(t)

    @property
    def nsteps(self):
        return self.nacptsteps + self.nrjctsteps

    def collect_stats(self, stats):
        wtime = time.time() - self._wstart

        # Rank allocation
        stats.set('backend', 'rank-allocation',
                  ','.join(str(r) for r in self.rallocs.mprankmap))

        # Simulation and wall clock times
        stats.set('solver-time-integrator', 'tcurr', self.tcurr)
        stats.set('solver-time-integrator', 'wall-time', wtime)

        # Step counts
        stats.set('solver-time-integrator', 'nsteps', self.nsteps)
        stats.set('solver-time-integrator', 'nacptsteps', self.nacptsteps)
        stats.set('solver-time-integrator', 'nrjctsteps', self.nrjctsteps)

    @property
    def cfgmeta(self):
        cfg = self.cfg.tostr()

        if self.prevcfgs:
            ret = dict(self.prevcfgs, config=cfg)

            if cfg != ret[f'config-{len(self.prevcfgs) - 1}']:
                ret[f'config-{len(self.prevcfgs)}'] = cfg

            return ret
        else:
            return {'config': cfg, 'config-0': cfg}

    def _check_abort(self):
        comm, rank, root = get_comm_rank_root()
        if comm.allreduce(self.abort, op=get_mpi('lor')):
            # Ensure that the callbacks registered in atexit
            # are called only once if stopping the computation
            sys.exit(1)


class BaseCommon:
    def _get_gndofs(self):
        comm, rank, root = get_comm_rank_root()

        # Get the number of degrees of freedom in this partition
        ndofs = sum(self.system.ele_ndofs)

        # Sum to get the global number over all partitions
        return comm.allreduce(ndofs, op=get_mpi('sum'))

    @memoize
    def _get_axnpby_kerns(self, *rs, subdims=None):
        kerns = [self.backend.kernel('axnpby', *[em[r] for r in rs],
                                     subdims=subdims)
                 for em in self.system.ele_banks]

        return kerns

    @memoize
    def _get_reduction_kerns(self, *rs, **kwargs):
        dtau_mats = getattr(self, 'dtau_upts', [])

        kerns = []
        for em, dtaum in it.zip_longest(self.system.ele_banks, dtau_mats):
            kerns.append(self.backend.kernel('reduction', *[em[r] for r in rs],
                                             dt_mat=dtaum, **kwargs))

        return kerns

    def _addv(self, consts, regidxs, subdims=None):
        # Get a suitable set of axnpby kernels
        axnpby = self._get_axnpby_kerns(*regidxs, subdims=subdims)

        # Bind and run the axnpby kernels
        self._queue.enqueue_and_run(axnpby, *consts)

    def _add(self, *args, subdims=None):
        self._addv(args[::2], args[1::2], subdims=subdims)
