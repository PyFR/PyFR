# -*- coding: utf-8 -*-

from pyfr.h5writer import H5Writer
from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin


class SolnWriterPlugin(BasePlugin):
    name = 'solnwriter'
    systems = ['*']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Construct the solution writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.')
        basename = self.cfg.get(cfgsect, 'basename', raw=True)
        self._writer = H5Writer(intg, basedir, basename, 'soln')

        # Output time step and next output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_next = intg.tcurr

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then write out the initial solution
        if not intg.isrestart:
            self(intg)
        else:
            self.tout_next += self.dt_out

    def __call__(self, intg):
        if abs(self.tout_next - intg.tcurr) > intg.dtmin:
            return

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        intg.collect_stats(stats)

        metadata = dict(config=self.cfg.tostr(),
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)

        self._writer.write(intg.soln, metadata, intg.tcurr)

        self.tout_next += self.dt_out
