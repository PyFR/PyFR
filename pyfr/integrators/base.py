# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict

from pyfr.inifile import Inifile
from pyfr.nputil import range_eval
from pyfr.util import proxylist


class BaseIntegrator(object):
    __metaclass__ = ABCMeta

    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        from mpi4py import MPI

        self._backend = backend
        self._rallocs = rallocs
        self._cfg = cfg

        # Sanity checks
        if self._controller_needs_errest and not self._stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Start time
        self.tstart = cfg.getfloat('solver-time-integrator', 't0', 0.0)

        # Output times
        self.tout = sorted(range_eval(cfg.get('soln-output', 'times')))
        self.tend = self.tout[-1]

        # Current time; defaults to tstart unless resuming a simulation
        if initsoln is None or 'stats' not in initsoln:
            self.tcurr = self.tstart
        else:
            stats = Inifile(initsoln['stats'])
            self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')

            # Cull already written output times
            self.tout = [t for t in self.tout if t > self.tcurr]

        # Ensure no time steps are in the past
        if self.tout[0] < self.tcurr:
            raise ValueError('Output times must be in the future')

        # Determine the amount of temp storage required by thus method
        nreg = self._stepper_nregs

        # Construct the relevant mesh partition
        self._system = systemcls(backend, rallocs, mesh, initsoln, nreg, cfg)

        # Extract the UUID of the mesh (to be saved with solutions)
        self._mesh_uuid = mesh['mesh_uuid'].item()

        # Get a queue for subclasses to use
        self._queue = backend.queue()

        # Get the number of degrees of freedom in this partition
        ndofs = sum(self._system.ele_ndofs)

        # Sum to get the global number over all partitions
        self._gndofs = MPI.COMM_WORLD.allreduce(ndofs, op=MPI.SUM)

    def _kernel(self, name, nargs):
        # Transpose from [nregs][neletypes] to [neletypes][nregs]
        transregs = zip(*self._regs)

        # Generate an kernel for each element type
        kerns = proxylist([])
        for tr in transregs:
            kerns.append(self._backend.kernel(name, *tr[:nargs]))

        return kerns

    def _prepare_reg_banks(self, *bidxes):
        for reg, ix in zip(self._regs, bidxes):
            reg.active = ix

    @abstractmethod
    def step(self, t, dt):
        pass

    @abstractmethod
    def advance_to(self, t):
        pass

    @abstractmethod
    def output(self, solns, stats):
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
        for t in self.tout:
            # Advance to time t
            solns = self.advance_to(t)

            # Map solutions to elements types
            solnmap = OrderedDict(zip(self._system.ele_types, solns))

            # Collect statistics
            stats = Inifile()
            self.collect_stats(stats)

            # Output
            self.output(solnmap, stats)

    def collect_stats(self, stats):
        stats.set('solver-time-integrator', 'tcurr', self.tcurr)
