# -*- coding: utf-8 -*-

import numpy as np

from pyfr.h5writer import H5Writer
from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin


class TavgPlugin(BasePlugin):
    name = 'tavg'
    systems = ['*']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        basedir = self.cfg.getpath(cfgsect, 'basedir', '.')
        basename = self.cfg.get(cfgsect, 'basename', raw=True)
        self._writer = H5Writer(intg, basedir, basename, 'soln')

        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self.nsteps = self.cfg.getfloat(cfgsect, 'nsteps')
        self.tout = intg.tcurr + self.dtout

        intg.call_plugin_dt(self.dtout)

        self.asoln = [np.zeros_like(s) for s in intg.soln]
        self.psoln = [s.copy() for s in intg.soln]
        self.pt = intg.tcurr

    def __call__(self, intg):
        dowrite = abs(self.tout - intg.tcurr) < intg.dtmin
        doaccum = intg.nacptsteps % self.nsteps == 0

        if dowrite or doaccum:
            for a, p, c in zip(self.asoln, self.psoln, intg.soln):
                a += 0.5*(intg.tcurr - self.pt)*(p + c)

            self.psoln = [s.copy() for s in intg.soln]
            self.pt = intg.tcurr

            if dowrite:
                self.asoln = [a / self.dtout for a in self.asoln]

                stats = Inifile()
                stats.set('tavg', 'start', intg.tcurr - self.dtout)
                stats.set('tavg', 'end', intg.tcurr)

                metadata = dict(config=self.cfg.tostr(),
                                stats=stats.tostr(),
                                mesh_uuid=intg.mesh_uuid)

                self._writer.write(self.asoln, metadata, intg.tcurr)

                self.tout += self.dtout
                self.asoln = [np.zeros_like(s) for s in intg.soln]
