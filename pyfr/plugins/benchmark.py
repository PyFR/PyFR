from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BaseSolnPlugin, init_csv
from pyfr.readers.native import NativeReader

class BenchmarkPlugin(BaseSolnPlugin):
    name = 'benchmark'
    systems = ['*']
    formulations = ['std']

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
            self.outf = init_csv(self.cfg, cfgsect, 'n,t,walldt,performance')
        else:
            self.outf = None

        mesh = NativeReader(self.cfg.get(self.cfgsect, 'mesh'))
        order = self.cfg.getint('solver', 'order')

        mesh_nep = mesh.partition_info('spt')

        # Number of degrees of freedom per element, depends on order
        # TODO: Get this by counting degrees() in polys.py
        dof_in_elem = {'hex': (order+1)**3,
                       'tet': (order+1)*(order+2)*(order+3)//6,
                        'pri': (order+1)*(order+2)*(order+3)//2,
                        'qua': (order+1)**2,
                        'pyr': (order+1)*(order+2)//2,
                        'tri': (order+1)*(order+2)//2}

        mesh_dof = [dof_in_elem[e] * sum(mesh_nep[e]) for e in mesh_nep.keys()]
        self.factor = sum(mesh_dof) * intg.stepper_order * intg.system.nvars

        self.mean = 0.0
        self.rem = 0.0

        # Allowed error in performance 
        self.tol = self.cfg.getfloat(self.cfgsect, 'tol', 1e-3)

    def __call__(self, intg):
        # Process the sequence of rejected/accepted steps
        for i, (walldt,) in enumerate(intg.perfinfo, start=self.count):

            perf = self.factor/walldt

            # Skip the first 12 steps
            if i >= 9:
                self.mean = (self.mean * (i-8) + perf) / (i-7)
                self.rem = (self.rem * (i-9) + (perf - self.mean)**2) / (i-8)
            else:
                self.mean = '-'
                self.rem = '-'

            self.stats.append((i, self.tprev, walldt, self.factor/walldt, self.mean, self.rem))

        # If self.rem is lesser than self.tol, then we have converged. 
        if self.rem < self.tol:
            raise RuntimeError(f'Converged at t = {intg.tcurr} with {self.rem} < {self.tol} after {intg.nacptsteps} steps')

        # Update the total step count and save the current time
        self.count += len(intg.perfinfo)
        self.tprev = intg.tcurr

        # If we're the root rank then output
        if self.outf:
            for s in self.stats:
                print(*s, sep=',', file=self.outf)

            # Periodically flush to disk
            if intg.nacptsteps % self.flushsteps == 0:
                self.outf.flush()

        # Reset the stats
        self.stats = []
