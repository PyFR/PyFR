# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import deque

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.util import proxylist


class BaseIntegrator(object, metaclass=ABCMeta):
    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        self.backend = backend
        self.rallocs = rallocs
        self.cfg = cfg
        self.isrestart = initsoln is not None

        # Sanity checks
        if self._controller_needs_errest and not self._stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Start time
        self.tstart = cfg.getfloat('solver-time-integrator', 'tstart', 0.0)
        self.tend = cfg.getfloat('solver-time-integrator', 'tend')

        # Current time; defaults to tstart unless restarting
        if self.isrestart:
            stats = Inifile(initsoln['stats'])
            self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
        else:
            self.tcurr = self.tstart

        self.tlist = deque([self.tend])

        # Determine the amount of temp storage required by thus method
        nreg = self._stepper_nregs

        # Construct the relevant mesh partition
        self.system = systemcls(backend, rallocs, mesh, initsoln, nreg, cfg)

        # Extract the UUID of the mesh (to be saved with solutions)
        self.mesh_uuid = mesh['mesh_uuid']

        # Get a queue for subclasses to use
        self._queue = backend.queue()

        # Get the number of degrees of freedom in this partition
        ndofs = sum(self.system.ele_ndofs)

        comm, rank, root = get_comm_rank_root()

        # Sum to get the global number over all partitions
        self._gndofs = comm.allreduce(ndofs, op=get_mpi('sum'))

    def _kernel(self, name, nargs):
        # Transpose from [nregs][neletypes] to [neletypes][nregs]
        transregs = zip(*self._regs)

        # Generate an kernel for each element type
        kerns = proxylist([])
        for tr in transregs:
            kerns.append(self.backend.kernel(name, *tr[:nargs]))

        return kerns

    def _prepare_reg_banks(self, *bidxes):
        for reg, ix in zip(self._regs, bidxes):
            reg.active = ix

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

    @abstractmethod
    def step(self, t, dt):
        pass

    @abstractmethod
    def advance_to(self, t):
        pass

    @abstractproperty
    def _controller_needs_errest(self):
        pass

    @abstractproperty
    def _stepper_has_errest(self):
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

    def collect_stats(self, stats):
        stats.set('solver-time-integrator', 'tcurr', self.tcurr)
