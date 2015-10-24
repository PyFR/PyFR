# -*- coding: utf-8 -*-

from pyfr.h5writer import H5Writer
from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin


class SolnWriterPlugin(BasePlugin):
    name = 'solnwriter'
    systems = ['*']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        basedir = self.cfg.getpath(cfgsect, 'basedir', '.')
        basename = self.cfg.get(cfgsect, 'basename', raw=True)
        self._writer = H5Writer(intg, basedir, basename, 'soln')

        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_next = intg.tcurr

        intg.call_plugin_dt(self.dt_out)

        # Call for writing the initial solution if not restart
        if intg.isrestart:
            self.tout_next += self.dt_out
        else:
            self(intg)

    def __call__(self, intg):
        if abs(self.tout_next - intg.tcurr) > intg.dtmin:
            return

        stats = Inifile()
        intg.collect_stats(stats)

        metadata = dict(config=self.cfg.tostr(),
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)

        self._writer.write(intg.soln, metadata, intg.tcurr)

        self.tout_next += self.dt_out
