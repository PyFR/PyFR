# -*- coding: utf-8 -*-

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin, init_csv


class DtStatsPlugin(BasePlugin):
    name = 'dtstats'
    systems = ['*']

    def __init__(self, intg, cfgsect, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 500)

        self.count = 0
        self.stats = []
        self.tprev = intg.tcurr

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # The root rank needs to open the output file
        if rank == root:
            self.outf = init_csv(self.cfg, cfgsect, 'n,t,dt,action,error')

    def __call__(self, intg):
        # Process the sequence of rejected/accepted steps
        for i, (dt, act, err) in enumerate(intg.stepinfo, start=self.count):
            self.stats.append((i, self.tprev, dt, act, err))

        # Update the total step count and save the current time
        self.count += len(intg.stepinfo)
        self.tprev = intg.tcurr

        comm, rank, root = get_comm_rank_root()

        # If we're the root rank then output
        if rank == root:
            for s in self.stats:
                print(','.join(str(c) for c in s), file=self.outf)

            # Periodically flush to disk
            if intg.nacptsteps % self.flushsteps == 0:
                self.outf.flush()

        # Reset the stats
        self.stats = []
