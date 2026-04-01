from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BaseSolnPlugin, init_csv


class DtStatsPlugin(BaseSolnPlugin):
    name = 'dtstats'
    systems = ['*']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.count = 0
        self.tprev = intg.tcurr
        self.stage_csv = self.step_csv = None

        comm, rank, root = get_comm_rank_root()

        # The root rank needs to open the output file(s)
        if rank == root:
            # Step file
            header = 'n,t,dt,action,wtime,error'
            self.step_csv = init_csv(self.cfg, cfgsect, header)

            # Stage file; optional, for implicit integrators only
            if (intg.formulation == 'implicit' and
                self.cfg.hasopt(cfgsect, 'stage-file')):
                header = ('n,stage,newton_iters,krylov_iters,precond_apps,'
                          'init_resid,final_resid,krylov_tol')
                self.stage_csv = init_csv(self.cfg, cfgsect, header,
                                          filekey='stage-file')

    def __call__(self, intg):
        # Only root rank writes output
        if self.step_csv:
            for i, info in enumerate(intg.stepinfo, start=self.count):
                if self.stage_csv and info.stages:
                    for s in info.stages:
                        self.stage_csv(i, *s)

                err = info.err or ''
                self.step_csv(i, self.tprev, info.dt, info.action, info.wtime,
                              err)

        # Update step count and time
        self.count += len(intg.stepinfo)
        self.tprev = intg.tcurr
