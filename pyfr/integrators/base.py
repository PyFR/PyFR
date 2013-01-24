# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty

from pyfr.inifile import Inifile
from pyfr.mesh_partition import get_mesh_partition
from pyfr.nputil import range_eval

class BaseIntegrator(object):
    __metaclass__ = ABCMeta

    def __init__(self, backend, rallocs, mesh, initsoln, cfg):
        self._backend = backend
        self._rallocs = rallocs
        self._cfg = cfg

        # Sanity checks
        if self._controller_needs_errest and not self._stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Start time
        self.tstart = cfg.getfloat('time-integration', 't0', 0.0)

        # Current time; defaults to tstart unless resuming a simulation
        if initsoln is None or 'stats' not in initsoln:
            self.tcurr = self.tstart
        else:
            stats = Inifile(initsoln['stats'])
            self.tcurr = stats.getfloat('time-integration', 'tcurr')

        # Output times
        self.tout = sorted(range_eval(cfg.get('soln-output', 'times')))
        self.tend = self.tout[-1]

        if self.tout[0] < self.tcurr:
            raise ValueError('Output times must be in the future')

        # Determine the amount of temp storage required by thus method
        nreg = self._stepper_nregs

        # Construct the relevant mesh partition
        self._meshp = get_mesh_partition(backend, rallocs, mesh, initsoln,
                                         nreg, cfg)

        # Extract the UUID of the mesh (to be saved with solutions)
        self._mesh_uuid = mesh['mesh_uuid'].item()

        # Get a queue for subclasses to use
        self._queue = backend.queue()

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

    def run(self):
        for t in self.tout:
            # Advance to time t
            solns = self.advance_to(t)

            # Map solutions to elements types
            solnmap = dict(zip(self._meshp.ele_types, solns))

            # Collect statistics
            stats = Inifile()
            self.collect_stats(stats)

            # Output
            self.output(solnmap, stats)

    def collect_stats(self, stats):
        stats.set('time-integration', 'tcurr', self.tcurr)
