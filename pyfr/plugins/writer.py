# -*- coding: utf-8 -*-

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin


class WriterPlugin(PostactionMixin, RegionMixin, BasePlugin):
    name = 'writer'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Construct the solution writer
        self._writer = self._init_writer_for_region(intg, self.nvars, 'soln')

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then write out the initial solution
        if not intg.isrestart:
            self.tout_last -= self.dt_out
            self(intg)

    def __call__(self, intg):
        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)

        # Prepare the metadata
        metadata = dict(intg.cfgmeta,
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)


        # Extract and subset the solution
        soln = [intg.soln[i][..., rgn] for i, rgn in self._ele_regions]

        # Add in any required region data
        data = self._add_region_data(soln)

        # Write out the file
        solnfname = self._writer.write(data, metadata, intg.tcurr)

        # If a post-action has been registered then invoke it
        self._invoke_postaction(mesh=intg.system.mesh.fname, soln=solnfname,
                                t=intg.tcurr)

        # Update the last output time
        self.tout_last = intg.tcurr
