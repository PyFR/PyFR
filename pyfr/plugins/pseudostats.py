# -*- coding: utf-8 -*-

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin, init_csv


class PseudoStatsPlugin(BasePlugin):
    name = 'pseudostats'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 500)

        self.count = 0
        self.stats = []
        self.tprev = intg.tcurr

        fvars = ','.join(intg.system.elementscls.convarmap[self.ndims])

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # The root rank needs to open the output file
        if rank == root:
            self.outf = init_csv(self.cfg, cfgsect, 'n,t,i,' + fvars)
        else:
            self.outf = None

    def __call__(self, intg):
        # Process the sequence of pseudo-residuals
        for (npiter, iternr, resid) in intg.pseudostepinfo:
            resid = resid or ('-',)*intg.system.nvars
            self.stats.append((npiter, self.tprev, iternr) + resid)

        # Update the total step count and save the current time
        self.count += len(intg.pseudostepinfo)
        self.tprev = intg.tcurr

        # If we're the root rank then output
        if self.outf:
            for s in self.stats:
                print(','.join(str(c) for c in s), file=self.outf)

            # Periodically flush to disk
            if intg.nacptsteps % self.flushsteps == 0:
                self.outf.flush()

        # Reset the stats
        self.stats = []
