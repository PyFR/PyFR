import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, init_csv


class ResidualPlugin(BaseSolnPlugin):
    name = 'residual'
    systems = ['*']
    formulations = ['std', 'dual']
    dimensions = [2, 3]
    
    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Norm used on residual
        self.lp = self.cfg.getfloat(cfgsect, 'norm', 2)

        # Set MPI reduction op and post process function
        if self.lp == float('inf'):
            self._mpi_op = mpi.MAX
            self._post_func = lambda x: x
        else:
            self._mpi_op = mpi.SUM
            self._post_func = lambda x: x**(1/self.lp)

        # The root rank needs to open the output file
        if rank == root:
            header = ['t'] + intg.system.elementscls.convarmap[self.ndims]

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

    def __call__(self, intg):
        # If an output is due this step
        if intg.nacptsteps % self.nsteps == 0 and intg.nacptsteps:
            # MPI info
            comm, rank, root = get_comm_rank_root()

            # Rank local norm for each variable
            norm = lambda x: np.linalg.norm(x, axis=(0, 2), ord=self.lp)
            if self.lp == float('inf'):
                resid = max(norm(dt_s) for dt_s in intg.dt_soln)
            else:
                resid = sum(norm(dt_s)**self.lp for dt_s in intg.dt_soln)

            # Reduce and, if we are the root rank, output
            if rank != root:
                comm.Reduce(resid, None, op=self._mpi_op, root=root)
            else:
                comm.Reduce(mpi.IN_PLACE, resid, op=self._mpi_op, root=root)

                # Post process
                resid = (self._post_func(r) for r in resid)

                # Write
                print(intg.tcurr, *resid, sep=',', file=self.outf)

                # Flush to disk
                self.outf.flush()
