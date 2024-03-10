from time import perf_counter

import numpy as np

from pyfr.mpiutil import mpi
from pyfr.plugins.base import BasePlugin

import csv

class ComputeTimePlugin(BasePlugin):
    name = 'computetime'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps', 1)

        intg.Δc = 0

        self.start_time = 0
        
    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:

            intg.compute_time = perf_counter() - self.start_time
            self.start_time = perf_counter()

            # output to csv
            #print(f"{intg.compute_time}, {intg.Δc}", file=self.outf, flush=True)
            with open('compute_time8.csv', mode='a') as compute_time_file:
                compute_time_writer = csv.writer(compute_time_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                compute_time_writer.writerow([
                    intg.compute_time,            
                    intg.Δc,
                    ])
