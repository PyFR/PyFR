# -*- coding: utf-8 -*-

import numpy as np

from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin
from pyfr.nputil import npeval
from pyfr.writers.native import NativeWriter


class TavgPlugin(BasePlugin):
    name = 'tavg'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Expressions to time average
        c = self.cfg.items_as('constants', float)
        self.exprs = [(k, self.cfg.getexpr(cfgsect, k, subs=c))
                      for k in self.cfg.items(cfgsect)
                      if k.startswith('avg-')]

        # Save a reference to the physical solution point locations
        self.plocs = intg.system.ele_ploc_upts

        # Output file directory, base name, and writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')
        self._writer = NativeWriter(intg, len(self.exprs), basedir, basename,
                                    prefix='tavg')

        # Time averaging parameters
        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        self.tout = intg.tcurr + self.dtout

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dtout)

        # Time averaging state
        self.prevt = intg.tcurr
        self.prevex = self._eval_exprs(intg)
        self.accmex = [np.zeros_like(p) for p in self.prevex]

    def _eval_exprs(self, intg):
        exprs = []

        # Iterate over each element type in the simulation
        for soln, ploc in zip(intg.soln, self.plocs):
            # Get the primitive variable names and solutions
            pnames = self.elementscls.privarmap[self.ndims]
            psolns = self.elementscls.con_to_pri(soln.swapaxes(0, 1),
                                                 self.cfg)

            # Prepare the substitutions dictionary
            ploc = dict(zip('xyz', ploc.swapaxes(0, 1)))
            subs = dict(zip(pnames, psolns), t=intg.tcurr, **ploc)

            # Evaluate the expressions
            exprs.append([npeval(v, subs) for k, v in self.exprs])

        # Stack up the expressions for each element type and return
        return [np.dstack(exs).swapaxes(1, 2) for exs in exprs]

    def __call__(self, intg):
        dowrite = abs(self.tout - intg.tcurr) < self.tol
        doaccum = intg.nacptsteps % self.nsteps == 0

        if dowrite or doaccum:
            # Evaluate the time averaging expressions
            currex = self._eval_exprs(intg)

            # Accumulate them; always do this even when just writing
            for a, p, c in zip(self.accmex, self.prevex, currex):
                a += 0.5*(intg.tcurr - self.prevt)*(p + c)

            # Save the time and solution
            self.prevt = intg.tcurr
            self.prevex = currex

            if dowrite:
                # Normalise
                accmex = [a / self.dtout for a in self.accmex]

                stats = Inifile()
                stats.set('data', 'prefix', 'tavg')
                stats.set('data', 'fields',
                          ','.join(k for k, v in self.exprs))
                stats.set('tavg', 'tstart', intg.tcurr - self.dtout)
                stats.set('tavg', 'tend', intg.tcurr)
                intg.collect_stats(stats)

                metadata = dict(intg.cfgmeta,
                                stats=stats.tostr(),
                                mesh_uuid=intg.mesh_uuid)

                self._writer.write(accmex, metadata, intg.tcurr)

                self.tout = intg.tcurr + self.dtout
                self.accmex = [np.zeros_like(a) for a in accmex]
