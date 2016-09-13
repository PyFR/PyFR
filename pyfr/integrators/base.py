# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import deque
import re
import time

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins import get_plugin
from pyfr.util import memoize, proxylist


class BaseIntegrator(object, metaclass=ABCMeta):
    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        self.backend = backend
        self.rallocs = rallocs
        self.isrestart = initsoln is not None
        self.cfg = cfg
        self.prevcfgs = {f: initsoln[f] for f in initsoln or []
                         if f.startswith('config-')}

        # Ensure the system is compatible with our formulation
        if self.formulation not in systemcls.elementscls.formulations:
            raise RuntimeError(
                'System {0} does not support time stepping formulation {1}'
                .format(systemcls.name, self.formulation)
            )

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
        self._dt = self.cfg.getfloat('solver-time-integrator', 'dt')
        self.dtmin = 1.0e-12

        # Determine the amount of temp storage required by this method
        nreg = self._stepper_nregs

        # Construct the relevant mesh partition
        self.system = systemcls(backend, rallocs, mesh, initsoln, nreg, cfg)

        # Storage register banks
        self._regs, self._regidx = self._get_reg_banks(nreg)

        # Extract the UUID of the mesh (to be saved with solutions)
        self.mesh_uuid = mesh['mesh_uuid']

        # Get a queue for subclasses to use
        self._queue = backend.queue()

        # Global degree of freedom count
        self._gndofs = self._get_gndofs()

        # Bank index of solution
        self._idxcurr = 0

        # Solution cache
        self._curr_soln = None

        # Add kernel cache
        self._axnpby_kerns = {}

        # Record the starting wall clock time
        self._wstart = time.time()

        # Event handlers for advance_to
        self.completed_step_handlers = proxylist(self._get_plugins())

        # Delete the memory-intensive elements map from the system
        del self.system.ele_map

    def _get_reg_banks(self, nreg):
        regs, regidx = [], list(range(nreg))

        # Create a proxylist of matrix-banks for each storage register
        for i in regidx:
            regs.append(
                proxylist([self.backend.matrix_bank(em, i)
                           for em in self.system.ele_banks])
            )

        return regs, regidx

    def _get_gndofs(self):
        comm, rank, root = get_comm_rank_root()

        # Get the number of degrees of freedom in this partition
        ndofs = sum(self.system.ele_ndofs)

        # Sum to get the global number over all partitions
        return comm.allreduce(ndofs, op=get_mpi('sum'))

    def _get_plugins(self):
        plugins = []

        for s in self.cfg.sections():
            m = re.match('soln-plugin-(.+?)(?:-(.+))?$', s)
            if m:
                cfgsect, name, suffix = m.group(0), m.group(1), m.group(2)

                # Instantiate
                plugins.append(get_plugin(name, self, cfgsect, suffix))

        return plugins

    def _get_kernels(self, name, nargs, **kwargs):
        # Transpose from [nregs][neletypes] to [neletypes][nregs]
        transregs = zip(*self._regs)

        # Generate an kernel for each element type
        kerns = proxylist([])
        for tr in transregs:
            kerns.append(self.backend.kernel(name, *tr[:nargs], **kwargs))

        return kerns

    def _prepare_reg_banks(self, *bidxes):
        for reg, ix in zip(self._regs, bidxes):
            reg.active = ix

    @memoize
    def _get_axnpby_kerns(self, n, subdims=None):
        return self._get_kernels('axnpby', nargs=n, subdims=subdims)

    def _add(self, *args):
        # Get a suitable set of axnpby kernels
        axnpby = self._get_axnpby_kerns(len(args) // 2)

        # Bank indices are in odd-numbered arguments
        self._prepare_reg_banks(*args[1::2])

        # Bind and run the axnpby kernels
        self._queue % axnpby(*args[::2])

    def call_plugin_dt(self, dt):
        ta = self.tlist
        tb = deque(np.arange(self.tcurr, self.tend, dt).tolist())

        self.tlist = tlist = deque()

        # Merge the current and new time lists
        while ta and tb:
            t = ta.popleft() if ta[0] < tb[0] else tb.popleft()
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

        tlist.extend(ta)
        tlist.extend(tb)

    @property
    def soln(self):
        # If we do not have the solution cached then fetch it
        if not self._curr_soln:
            self._curr_soln = self.system.ele_scal_upts(self._idxcurr)

        return self._curr_soln

    @abstractmethod
    def step(self, t, dt):
        pass

    @abstractmethod
    def advance_to(self, t):
        pass

    @abstractproperty
    def _stepper_nfevals(self):
        pass

    @abstractproperty
    def _stepper_nregs(self):
        pass

    @abstractproperty
    def _stepper_order(self):
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

            if cfg != ret['config-' + str(len(self.prevcfgs) - 1)]:
                ret['config-' + str(len(self.prevcfgs))] = cfg

            return ret
        else:
            return {'config': cfg, 'config-0': cfg}
