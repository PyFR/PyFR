import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, init_csv
from pyfr.util import first


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

        # Reduction parameters
        if self.lp == float('inf'):
            self._lp_exp = 1
            self._np_op = np.maximum
            self._mpi_op = mpi.MAX
        else:
            self._lp_exp = self.lp
            self._np_op = np.add
            self._mpi_op = mpi.SUM

        # The root rank needs to open the output file
        if rank == root:
            header = ['t'] + first(intg.system.ele_map.values()).convars

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

    def __call__(self, intg):
        # If an output is due this step
        if intg.nacptsteps % self.nsteps == 0 and intg.nacptsteps:
            # MPI info
            comm, rank, root = get_comm_rank_root()

            # Compute the norms of du/dt
            norms = []
            for dudt in intg.dt_soln:
                dudt = dudt.swapaxes(0, 1).reshape(self.nvars, -1)
                norms.append(np.linalg.norm(dudt, axis=1, ord=self.lp))

            # Reduce over each element type in our domain
            resid = self._np_op.reduce([n**self._lp_exp for n in norms])

            # Reduce over all domains and, if we are the root rank, output
            if rank != root:
                comm.Reduce(resid, None, op=self._mpi_op, root=root)
            else:
                comm.Reduce(mpi.IN_PLACE, resid, op=self._mpi_op, root=root)

                # Post process
                resid = (r**(1 / self._lp_exp) for r in resid)

                # Write
                print(intg.tcurr, *resid, sep=',', file=self.outf)

                # Flush to disk
                self.outf.flush()
