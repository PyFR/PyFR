# -*- coding: utf-8 -*-

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter


class WriterPlugin(PostactionMixin, RegionMixin, BasePlugin):
    name = 'writer'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Construct the solution writer
        self._writer = NativeWriter(intg, basedir, basename, 'soln')

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Output data type
        self.fpdtype = intg.backend.fpdtype

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then make sure we write out the initial
        # solution when we are called for the first time
        if not intg.isrestart:
            self.tout_last -= self.dt_out

    def __call__(self, intg):
        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

        comm, rank, root = get_comm_rank_root()

        # If we are the root rank then prepare the metadata
        if rank == root:
            stats = Inifile()
            stats.set('data', 'fields', ','.join(self.fields))
            stats.set('data', 'prefix', 'soln')
            intg.collect_stats(stats)

            metadata = dict(intg.cfgmeta,
                            stats=stats.tostr(),
                            mesh_uuid=intg.mesh_uuid)
        else:
            metadata = None

        # Fetch data from other plugins and add it to metadata with ad-hoc keys
        for csh in intg.completed_step_handlers:
            try:
                prefix = intg.get_plugin_data_prefix(csh.name, csh.suffix)
                pdata = csh.serialise(intg)
            except AttributeError:
                pdata = {}

            if rank == root:
                metadata |= {f'{prefix}/{k}': v for k, v in pdata.items()}

        # Fetch and (if necessary) subset the solution
        data = dict(self._ele_region_data)
        for idx, etype, rgn in self._ele_regions:
            data[etype] = intg.soln[idx][..., rgn].astype(self.fpdtype)

        # Write out the file
        solnfname = self._writer.write(data, intg.tcurr, metadata)

        # If a post-action has been registered then invoke it
        self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                soln=solnfname, t=intg.tcurr)

        # Update the last output time
        self.tout_last = intg.tcurr
