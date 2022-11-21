import os
from time import time

import numpy as np
import pandas as pd

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin

dtyp = np.float64

class OptimisationStatsPlugin(BasePlugin):
    name = 'optimisation_stats'
    systems = ['*']
    formulations = ['dual']
    
    def __init__(self, intg, cfgsect, suffix):
        
        self.comm, self.rank, self.root = get_comm_rank_root()
        super().__init__(intg, cfgsect, suffix)
        
        self.tstart = self.cfg.getfloat(cfgsect, 'tstart', 0.0)     # Start collecting stats
        self.tend   = self.cfg.getfloat('solver-time-integrator', 'tend', 0.0)     # Start collecting stats
        

        self.skip_first_n   = self.cfg.getint(cfgsect,   'skip_first_n', 1)     # Skip first n iterations
        self.capture_last_n = self.cfg.getint(cfgsect, 'capture_last_n', 3)     # Capture last n iterations


        self.outf  = self.cfg.get(cfgsect, 'out-file-expanded' , 'pre-opt_exp.csv')
        self.outf2 = self.cfg.get(cfgsect, 'out-file-condensed', 'pre-opt_con.csv')


        if self.rank == self.root:

            self.opt_time      = 0.
            self.bounds_length = 0.

            self.maxniters = self.cfg.getint('solver-time-integrator', 'pseudo-niters-max')
            self.Δτ_controller = self.cfg.get('solver-time-integrator', 'pseudo-controller')

            self.Δτ_init = self.cfg.getfloat('solver-time-integrator', 'pseudo-dt')

            if self.Δτ_controller == 'local-pi':
                self.Δτ_max = self.Δτ_init *  self.cfg.getfloat('solver-time-integrator', 'pseudo-dt-max-mult')

            self.pd_stats =  pd.DataFrame()

            self.ctime_p, self.wtime_p = 0, 0
            self.reset_stats()

        intg.reset_opt_stats = False

    def __call__(self, intg):

        # Collect stats after tstart
        if self.tstart > intg.tcurr:
            return

        # Collect stats from the integrator
        if self.rank == self.root:

            # If optimiser has used the last bastch of data, then clean slate.
            if intg.reset_opt_stats == True:

                if os.path.exists(self.outf):
                    self.pd_stats.to_csv(self.outf, header=False, index=False, mode='a')
                    self.pd_stats.tail(1).to_csv(self.outf2, header=False, index=False, mode='a')
                else:
                    self.pd_stats.to_csv(self.outf, header=True, index=False, mode='w')
                    self.pd_stats.tail(1).to_csv(self.outf2, header=True, index=False, mode='w')

                self.reset_stats()
                intg.reset_opt_stats = False

            self.collect_stats(intg)
            self.check_status(intg)

            if self.tend < intg.tcurr:
                self.print_stats()

    def print_stats(self):
        print(self.pd_stats)

    def reset_stats(self):
        self.pd_stats = pd.DataFrame()

    def collect_stats(self, intg) -> None:
        Δt = intg._dt  # intg.tcurr - self.ttime_p                # Physical time step
        Δc = intg.pseudointegrator._compute_time - self.ctime_p  # Compute time per physical time-step
        # Wall-time per physical time-step
        Δw = (time() - intg._wstart) - self.wtime_p
        self.ΔcΔt, ΔwΔt = Δc/Δt, Δw/Δt

        self.wtime_p = time() - intg._wstart                # previous   Wall   time
        self.ctime_p = intg.pseudointegrator._compute_time  # previous Compute  time

        #temp = pd.DataFrame([[intg.tcurr, Δc, Δw]], columns=['physical-time', 'compute-Δt', 'wall-Δt'])

        t1 = pd.DataFrame({'physical-time': [intg.tcurr - intg._dt], 
                                'compute-Δt'   : [Δc], 
                                'wall-Δt'      : [Δw],
                                'cost'         : [self.ΔcΔt]}
                            )

        # Here, if intg data from plugin exists, we take data from there 
        if self.Δτ_controller == 'local-pi' and intg.Δτ_stats:
            t1['max-Δτ'] = intg.Δτ_stats['max']['all']
            t1['min-Δτ'] = intg.Δτ_stats['min']['all']
            t1['n']      = intg.Δτ_stats[ 'n' ]['all']
        elif self.Δτ_controller == 'none':
            t1['Δτ'] = self.Δτ_init

        t1['invalid'] = None
        self.pd_stats = pd.concat([self.pd_stats, t1], ignore_index=True)

        self.pd_stats = self.pd_stats.assign(
            **{  'cost_mean': self.pd_stats[  'cost'].rolling(5, min_periods=1).mean(),
                })

    def check_status(self, intg):

        # if .pyfrs file was not given at start, the simulation is probably not supposed to be optimised.

        # Stop because simulation is totally bad
        if any(np.isnan(np.sum(s)) for s in intg.soln):
            self.pd_stats.at[self.pd_stats.index[-1], 'invalid'] = 21
            # Invalidate part of the data in the dataframe
            return

        if self.pd_stats.count(0)[0]<=3:
            self.pd_stats.at[self.pd_stats.index[-1], 'invalid'] = 0
            return

        if self.pd_stats['n'][self.pd_stats.index[-1]] == self.maxniters*intg.nstages:
            self.pd_stats.at[self.pd_stats.index[-1], 'invalid'] = 22
            # Invalidate part of the data in the dataframe
            return

        # Simulation data should be calculated after we are sure that time-step will not change more than this
        if self.Δτ_controller == 'local-pi' and \
            -1e-6<self.pd_stats['max-Δτ'][self.pd_stats.index[-1]] - self.pd_stats['min-Δτ'][self.pd_stats.index[-1]]<1e-6:
                self.pd_stats.at[self.pd_stats.index[-1], 'invalid'] = 11       # Bad initial guess for Δτ
                return
        
        if self.Δτ_controller == 'none' or \
            (self.Δτ_controller == 'local-pi' and -1e-6<self.Δτ_max - self.pd_stats['max-Δτ'][self.pd_stats.index[-1]]<1e-6):

            if self.pd_stats.count(0)[0]> (self.skip_first_n + self.capture_last_n):
                self.pd_stats.at[self.pd_stats.index[-1], 'invalid'] = 1
                intg.reset_opt_stats = True
                return

        self.pd_stats.at[self.pd_stats.index[-1], 'invalid'] = 0
        return
