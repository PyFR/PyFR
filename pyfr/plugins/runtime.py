# -*- coding: utf-8 -*-

import os
import sys
import time

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin


class RuntimePlugin(PostactionMixin, RegionMixin, BasePlugin):
    name = 'runtime'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        intg.cfg.set(cfgsect, 'region', '*')
        super().__init__(intg, cfgsect, suffix)

        # Construct the solution writer
        self._writer = self._init_writer_for_region(intg, self.nvars, 'soln')

        # Max physical time in seconds
        self._max_phystime = self.cfg.getfloat(cfgsect, 'max-phystime', -1)

        # Max wall time in hours
        self._max_walltime = self.cfg.getfloat(cfgsect, 'max-walltime', -1)

        # Stop file
        self._stop_file = self.cfg.get(self.cfgsect, 'stop-file', '')

        # Check every nsteps
        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps', 50)

        # User needs to give at least one condition
        if self._max_phystime == -1 and self._max_walltime == -1 and \
           self._stop_file == '':
            raise RuntimeError("max-phystime, max_walltime "
                               "and stop-file not found in the input file")

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

    def __call__(self, intg):
        # Only check each nsteps
        if intg.nacptsteps % self.nsteps != 0:
            return

        # If max_phystime condition exists, check it
        if self._max_phystime != -1:
           if self._max_phystime > intg.tcurr:
               # Write all the solution data
               # and finalize the simulation
               self._write_exit(intg)

        # If max_phystime condition is not met
        # and max_walltime is not set
        # there is no need for further checks
        if self._max_walltime == -1:
            return

        # Obtain the current wall clock time
        # Ensure that all ranks utilize the same value
        # to avoid possible synchronization issues
        comm, rank, root = get_comm_rank_root()
        t = comm.bcast(time.time() if rank == 0 else None, root=0)
        wtime = t - intg._wstart

        # If the total simulation wall clock time is bigger than
        # max_walltime, write all the solution data and
        # finalize the simulation
        if wtime > self._max_walltime*60*24:
            self._write_exit(intg)

        # If a stop-file has been given by the user
        if self._stop_file != '':
            # Check if stop-file exists 
            # only in the first rank
            file_exists = comm.bcast(os.path.exists(
                                     os.path.join(self._writer.basedir, self._stop_file))
                                     if rank == 0 else None, root=0)
            if file_exists:
                # Write all the solution data
                # and finalize the simulation
                self._write_exit(intg)

    def _write_exit(self, intg):
        # Create Inifile for _writer
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

        # Ensure that the callbacks registered in atexit
        # are called only once
        sys.exit()
