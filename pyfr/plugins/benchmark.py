import numpy as np

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

        self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 1)

        self.count = 0
        self.stats = []
        self.tprev = intg.tcurr

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # The root rank needs to open the output file
        if rank == root:
            self.outf = init_csv(self.cfg, cfgsect, 'n,t,walldt,performance,mean,rel-err')
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
        self.var = 0.0

        # Allowed error in performance 
        self.tol = self.cfg.getfloat(self.cfgsect, 'tol', 1e-3)

        # If user wishes to continue simulation without any pause
        self.continue_sim = self.cfg.getbool(self.cfgsect, 'continue-sim', True)

        self.skip_first_n = self.cfg.getint(self.cfgsect, 'skip-first-n', 10)

    def __call__(self, intg):
        # Process the sequence of rejected/accepted steps
        for i, (walldt,) in enumerate(intg.perfinfo, start=self.count):

            perf = self.factor/walldt
            relative_error = 0.0
            
            # Skip the first 12 steps
            if i > self.skip_first_n:
                self.old_mean = self.mean
                self.mean = (self.mean * (i-self.skip_first_n-1) + perf) / (i-self.skip_first_n)

            if i > self.skip_first_n+1:
                self.var  = (self.var  * (i-self.skip_first_n-1) + (perf - self.old_mean)**2) / (i-self.skip_first_n)
                relative_error = np.sqrt(self.var)/self.mean
                
            self.stats.append((i, self.tprev, walldt, self.factor/walldt, self.mean, relative_error))

            # If self.var is lesser than self.tol, then we have converged. 
            if relative_error < self.tol and i > self.skip_first_n+2 and not self.continue_sim:
                raise RuntimeError(f'Converged at t = {intg.tcurr} with {self.var} < {self.tol} after {intg.nacptsteps} steps')

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
