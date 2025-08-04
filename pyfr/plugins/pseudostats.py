from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BaseSolnPlugin, init_csv
from pyfr.util import first


class PseudoStatsPlugin(BaseSolnPlugin):
    name = 'pseudostats'
    systems = ['*']
    formulations = ['dual']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.count = 0
        self.stats = []
        self.tprev = intg.tcurr

        fvars = ','.join(first(intg.system.ele_map.values()).convars)

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # The root rank needs to open the output file
        if rank == root:
            nflush = self.cfg.getint(self.cfgsect, 'flushsteps', 500)
            self.csv = init_csv(self.cfg, cfgsect, 'n,t,i,' + fvars, 
                                nflush=nflush)
        else:
            self.csv = None

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
                self.csv.print(*s, sep=',')

        # Reset the stats
        self.stats = []
