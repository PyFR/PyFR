# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty

from pyfr.inifile import Inifile
from pyfr.mesh_partition import MeshPartition
from pyfr.util import range_eval

class BaseIntegrator(object):
    __metaclass__ = ABCMeta

    def __init__(self, backend, rallocs, mesh, cfg):
        self._backend = backend
        self._rallocs = rallocs
        self._cfg = cfg

        # Sanity checks
        if self._controller_needs_errest and not self._stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Current times and output times
        self._tcurr = cfg.getfloat('time-integration', 't0', 0.0)
        self._tout = range_eval(cfg.get('soln-output', 'times'))

        # Determine the amount of temp storage required by thus method
        nreg = self._stepper_nregs

        # Construct the mesh partition
        self._meshp = MeshPartition(backend, rallocs, mesh, nreg, cfg)

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
        for t in self._tout:
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
        stats.set('time-integration', 'tcurr', self._tcurr)
