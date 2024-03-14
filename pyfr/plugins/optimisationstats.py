import os
from time import time

import numpy as np
import pandas as pd

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BaseSolnPlugin

class OptimisationStatsPlugin(BaseSolnPlugin):
    name = 'optimisation_stats'
    systems = ['*']
    formulations = ['dual']
    dimensions = [2, 3]
    
    def __init__(self, intg, cfgsect, suffix):
        
        self.comm, self.rank, self.root = get_comm_rank_root()
        super().__init__(intg, cfgsect, suffix)

        tsect = 'solver-time-integrator'
        
        # Start and stop collecting stats
        self.opt_tstart = self.cfg.getfloat(cfgsect, 'tstart', intg.tstart)
        intg.opt_tend = self.cfg.getfloat(cfgsect, 'tend', intg.tend)

        # Skip first few iterations, and capture the rest few iterations

        ref_window = self.cfg.getint(cfgsect, 'ref-window',  8)

        intg._increment         = ref_window//4
        intg._skip_first_n      = ref_window//2      
        intg._capture_next_n    = ref_window    

        intg._precision = self.cfg.getfloat(cfgsect, 'precision', 0.7)

        self.Δτ_init = self.cfg.getfloat(tsect, 'pseudo-dt')
        self.Δτ_controller = self.cfg.get(tsect, 'pseudo-controller')

        intg.reset_opt_stats = intg.bad_sim = False
        intg.opt_cost_mean = intg.opt_cost_sem = intg.opt_cost_std = None 

        if self.rank == self.root:

            if self.cfg.hasopt(cfgsect, 'file-expanded'):
                self.outf = self.cfg.get(cfgsect, 'file-expanded')
            else:
                self.outf = None

            if self.cfg.hasopt(cfgsect, 'file-condensed'):  
                self.outf2 = self.cfg.get(cfgsect, 'file-condensed')
            else:
                self.outf2 = None
    
            self.fvars = intg.system.elementscls.convarmap[self.ndims]

            if self.cfg.hasopt('solver-dual-time-integrator-multip', 'cycle'):
                self.maxniters = intg.pseudointegrator.pintg.maxniters
                self.minniters = intg.pseudointegrator.pintg.minniters
                self.residtols = intg.pseudointegrator.pintg._pseudo_residtol_l2
            else:
                self.maxniters = intg.pseudointegrator.maxniters
                self.minniters = intg.pseudointegrator.minniters
                self.residtols = intg.pseudointegrator._pseudo_residtol_l2

            self.ctime_p, self.wtime_p = 0, 0
            self.pd_stats = pd.DataFrame()
            self.pd_condensed_stats = pd.DataFrame()

    def __call__(self, intg):

        if intg.tcurr < self.opt_tstart:
            intg.pseudointegrator._compute_time = 0
            return

        # Collect stats from the integrator
        if self.rank == self.root:

            # If optimiser has used the data
            if intg.reset_opt_stats == True:

                if (self.outf2 is not None) and (intg.tcurr <= intg.opt_tend):
                    self.print_condensed_stats()
                if self.outf is not None:
                    self.print_expanded_stats()

                # Clean slate
                self.pd_stats = pd.DataFrame()
                intg.reset_opt_stats = False

            self.collect_stats(intg)
            self.check_status(intg)
        self.bcast_status(intg)

    def print_condensed_stats(self):
        self.pd_condensed_stats = pd.concat([self.pd_condensed_stats, 
                                             self.pd_stats.tail(1)], 
                                             ignore_index = True)
        self.pd_condensed_stats.to_csv(self.outf2, index = False)

    def print_expanded_stats(self):
        if os.path.exists(self.outf):
            self.pd_stats.to_csv(self.outf, header=False, index=False, mode='a')
        else:
            self.pd_stats.to_csv(self.outf, header=True, index=False, mode='w')

    def collect_stats(self, intg):
        # Physical time step
        Δt = intg._dt               
        # Compute time per physical time-step
        Δc = intg.pseudointegrator._compute_time - self.ctime_p  
        intg.Δc = Δc
        # Wall-time per physical time-step
        Δw = (time() - intg._wstart) - self.wtime_p
        self.ΔcΔt, ΔwΔt = Δc/Δt, Δw/Δt

        self.wtime_p = time() - intg._wstart                # prev   wall  time
        self.ctime_p = intg.pseudointegrator._compute_time  # prev compute time

        t1 = pd.DataFrame({ 'physical-time': [intg.tcurr - intg._dt], 
                            'compute-Δt'   : [Δc], 
                               'wall-Δt'   : [Δw],
                            'cost'         : [self.ΔcΔt],
                          }) 

        # Here, if intg data from plugin exists, we take data from there 
        if self.Δτ_controller == 'local-pi' and intg.dtau_stats:
            t1['max-Δτ'] = intg.dtau_stats['max']['all']
            t1['min-Δτ'] = intg.dtau_stats['min']['all']
            t1[     'n'] = intg.dtau_stats[ 'n' ]['all']
        elif self.Δτ_controller == 'none':
            t1['Δτ'] = self.Δτ_init

        self.pd_stats = pd.concat([self.pd_stats, t1], ignore_index=True)

    def check_status(self, intg):
        intg.actually_captured=self.pd_stats.count(0).iloc[0] - intg._skip_first_n

        # Stop collection if simulation explodes
        if any(np.isnan(np.sum(s)) for s in intg.soln):
            intg.actually_captured = -self.pd_stats.count(0)[0]
            intg.reset_opt_stats = intg.bad_sim = True
            intg.opt_cost_mean = intg.opt_cost_std = intg.opt_cost_sem = np.NaN 
            return

        # Continue if not enough data
        if self.pd_stats.count(0).iloc[0] < intg._increment:
            return

        # Stop collection if max-n-iterations is reached
        if (self.pd_stats['n'][self.pd_stats.index[-1]] == self.maxniters*intg.nstages): 
            if (self.maxniters != self.minniters):
                intg.actually_captured = -self.pd_stats.count(0)[0]
                intg.reset_opt_stats = intg.bad_sim = True
                intg.opt_cost_mean = intg.opt_cost_std = intg.opt_cost_sem = np.NaN
                return
       
        # Typical case: If Δτ is constant and enough data is collected
        if ((self.Δτ_controller == 'none'
              or (self.Δτ_controller == 'local-pi'
                  and abs(intg.pseudointegrator.pintg.Δτᴹ
                          - self.pd_stats['max-Δτ'][self.pd_stats.index[-1]]
                         )<1e-6
                  )
             )
            and self.pd_stats.count(0).iloc[0] >= (intg._skip_first_n + intg._capture_next_n)):

                intg.reset_opt_stats = True

                # Accumilate mean, std and sem after skipping steps
                intg.opt_cost_mean= self.pd_stats['cost'].tail(intg.actually_captured).mean()
                intg.opt_cost_std = self.pd_stats['cost'].tail(intg.actually_captured).std()
                intg.opt_cost_sem = self.pd_stats['cost'].tail(intg.actually_captured).sem()
                intg.bad_sim = False # (intg.opt_cost_std/intg.opt_cost_mean) > intg._precision

    def bcast_status(self, intg):
        intg.reset_opt_stats = self.comm.bcast(intg.reset_opt_stats, root=0)
        intg.bad_sim = self.comm.bcast(intg.bad_sim, root=0)
        