from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BaseSolnPlugin, init_csv


class DtStatsPlugin(BaseSolnPlugin):
    name = 'dtstats'
    systems = ['*']
    formulations = ['std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.count = 0
        self.stats = []
        self.tprev = intg.tcurr

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # The root rank needs to open the output file
        if rank == root:
            header = 'n,t,dt,action,error'
            self.csv = init_csv(self.cfg, cfgsect, header, nflush=500)
        else:
            self.csv = None

    def __call__(self, intg):
        # Process the sequence of rejected/accepted steps
        for i, (dt, act, err) in enumerate(intg.stepinfo, start=self.count):
            self.stats.append((i, self.tprev, dt, act, err))

        # Update the total step count and save the current time
        self.count += len(intg.stepinfo)
        self.tprev = intg.tcurr

        # If we're the root rank then output
        if self.csv:
            for s in self.stats:
                self.csv(*s)

        # Reset the stats
        self.stats = []
