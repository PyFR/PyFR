# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from pyfr.plugins.base import BasePlugin


class AveragePlugin(BasePlugin):
    name = 'average'
    systems = ['*']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')
        self.writeskip = self.cfg.getint(self.cfgsect, 'writeskip')
        self.tout = list(args[0].tout)
        self.averages = []
        self.solnprev = None
        self.tprev = None
        self.dtprev = 0
        self.prevwrite = args[0].tcurr
        self.avglist = OrderedDict()
        for (e, p), (name, shape) in args[0].sollist.items():
            self.avglist[e + '_avg', p] = (name + '_avg', shape)

    def __call__(self, intg):
        if not len(self.tout):
            return

        if (intg.nacptsteps % self.nsteps == 0) or\
           (intg.tcurr >= self.tout[0]):
            # if this is not the first iteration, we can integrate
            if self.tprev is not None:
                dt = intg.tcurr - self.tprev
                for avg, soln in zip(self.averages, self.solnprev):
                    avg += 0.5*(self.dtprev+dt)*soln

                self.dtprev = dt
                # if its time to write
                if intg.tcurr >= self.tout[0]:
                    for avg, soln in zip(self.averages, intg.soln):
                        avg += 0.5*dt*soln
                    avgnames = list(e + '_avg' for e in intg.system.ele_types)
                    path = intg._get_output_path().\
                        replace('.pyfrs', '_avg.pyfrs')

                    avgmap = OrderedDict(zip(avgnames, self.averages))
                    metadata = {'tstart': str(self.prevwrite),
                                'tend': str(self. tout[0])}

                    intg._write(path, avgmap, self.avglist, metadata)

                    self.prevwrite = self.tout.pop(0)
                    if self.writeskip:
                        self.tout = self.tout[self.writeskip:]
                    self.dtprev = 0
                    self.averages = [np.zeros_like(s) for s in intg.soln]
            else:
                self.averages = [np.zeros_like(s) for s in intg.soln]
            self.solnprev = intg.soln
            self.tprev = intg.tcurr
