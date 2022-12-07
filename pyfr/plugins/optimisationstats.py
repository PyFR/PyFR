import os
from time import time

import numpy as np
import pandas as pd

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin

class OptimisationStatsPlugin(BasePlugin):
    name = 'optimisation_stats'
    systems = ['*']
    formulations = ['dual']
    
    def __init__(self, intg, cfgsect, suffix):
        
        self.comm, self.rank, self.root = get_comm_rank_root()
        super().__init__(intg, cfgsect, suffix)

        tsect = 'solver-time-integrator'
        
        # Start and stop collecting stats
        self.tstart = self.cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.tend   = self.cfg.getfloat(tsect, 'tend', 0.0)

        # Skip first few iterations, and capture the rest few iterations
        self.skip_first_n = self.cfg.getint(cfgsect,   'skip-first-n', 10)     
        self.lastₙ = self.cfg.getint(cfgsect, 'capture-last-n', 30)

        self.Δτ_init       = self.cfg.getfloat(tsect, 'pseudo-dt')
        self.Δτ_controller = self.cfg.get(     tsect, 'pseudo-controller')
        if self.Δτ_controller == 'local-pi':
            self.Δτ_max = self.Δτ_init * self.cfg.getfloat(tsect, 'pseudo-dt-max-mult')

        intg.reset_opt_stats = intg.bad_sim = False
        intg.opt_cost_mean = intg.opt_cost_std = None 

        if self.rank == self.root:

            self.outf  = self.cfg.get(cfgsect, 'file-expanded', None)
            self.outf2 = self.cfg.get(cfgsect, 'file-condensed', 'o-stat_c.csv')

            self.fvars   = intg.system.elementscls.convarmap[self.ndims]

            if self.cfg.hasopt('solver-dual-time-integrator-multip', 'cycle'):
                self.maxniters = intg.pseudointegrator.pintg.maxniters
                self.minniters = intg.pseudointegrator.pintg.minniters
                self.residtols = intg.pseudointegrator.pintg._pseudo_residtol
            else:
                self.maxniters = intg.pseudointegrator.maxniters
                self.minniters = intg.pseudointegrator.minniters
                self.residtols = intg.pseudointegrator._pseudo_residtol

            self.ctime_p, self.wtime_p = 0, 0
            self.pd_stats = pd.DataFrame()
            self.pd_condensed_stats = pd.DataFrame()

    def __call__(self, intg):

        # Collect stats after tstart
        if self.tstart > intg.tcurr:
            return

        # Collect stats from the integrator
        if self.rank == self.root:

            # If optimiser has used the data
            if intg.reset_opt_stats == True:


                if self.outf2 is not None:

                    self.pd_condensed_stats = pd.concat([self.pd_condensed_stats, self.pd_stats.tail(1)], 
                                                        ignore_index = True)
                    self.pd_condensed_stats.to_csv(self.outf2, index = False)

                    self.print_condensed_stats()

                if self.outf != None:
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
        Δt = intg._dt  # intg.tcurr - self.ttime_p                # Physical time step
        Δc = intg.pseudointegrator._compute_time - self.ctime_p  # Compute time per physical time-step
        # Wall-time per physical time-step
        Δw = (time() - intg._wstart) - self.wtime_p
        self.ΔcΔt, ΔwΔt = Δc/Δt, Δw/Δt

        self.wtime_p = time() - intg._wstart                # previous   Wall   time
        self.ctime_p = intg.pseudointegrator._compute_time  # previous Compute  time

        #temp = pd.DataFrame([[intg.tcurr, Δc, Δw]], columns=['physical-time', 'compute-Δt', 'wall-Δt'])

        t1 = pd.DataFrame({ 'physical-time': [intg.tcurr - intg._dt], 
                            'compute-Δt'   : [Δc], 
                            'wall-Δt'      : [Δw],
                            'cost'         : [self.ΔcΔt],
                          }) 

        # Here, if intg data from plugin exists, we take data from there 
        if self.Δτ_controller == 'local-pi' and intg.Δτ_stats:
            t1['max-Δτ'] = intg.Δτ_stats['max']['all']
            t1['min-Δτ'] = intg.Δτ_stats['min']['all']
            t1['n']      = intg.Δτ_stats[ 'n' ]['all']
        elif self.Δτ_controller == 'none':
            t1['Δτ'] = self.Δτ_init

        self.pd_stats = pd.concat([self.pd_stats, t1], ignore_index=True)

        self.pd_stats = self.pd_stats.assign(
            **{'cost-m': self.pd_stats['cost']
                            .rolling(self.lastₙ, min_periods=1)
                            .mean(),
               'cost-s': self.pd_stats['cost']
                            .rolling(self.lastₙ, min_periods=1)
                            .std(),
                })

    def check_status(self, intg):

        # Stop because simulation is totally bad
        if any(np.isnan(np.sum(s)) for s in intg.soln):
            intg.reset_opt_stats = intg.bad_sim = True
            intg.opt_cost_mean = intg.opt_cost_std  = np.NaN 
            return

        if self.pd_stats.count(0)[0]<=3:
            return

        if (self.pd_stats['n'][self.pd_stats.index[-1]] == self.maxniters*intg.nstages): 
            if (self.maxniters != self.minniters):
                intg.reset_opt_stats = intg.bad_sim = True
                intg.opt_cost_mean = intg.opt_cost_std = np.NaN
                return

        if (self.pd_stats['n'][self.pd_stats.index[-1]] == self.maxniters*intg.nstages): 
            if np.any([intg.Δτ_stats['res'][var]['all'] > tol for var, tol in zip(self.fvars, self.residtols)]):
                intg.reset_opt_stats = intg.bad_sim = True
                intg.opt_cost_mean = intg.opt_cost_std = np.NaN
                return

        # Simulation data should be calculated after we are sure that time-step will not change more than this
        if (self.Δτ_controller == 'local-pi' and
            -1e-6<self.pd_stats['max-Δτ'][self.pd_stats.index[-1]] - self.pd_stats['min-Δτ'][self.pd_stats.index[-1]]<1e-6):
            intg.reset_opt_stats = intg.bad_sim = True
            intg.opt_cost_mean = intg.opt_cost_std = np.NaN
        
        if (((self.Δτ_controller == 'none'
             or (self.Δτ_controller == 'local-pi'
            and abs(self.Δτ_max - self.pd_stats['max-Δτ'][self.pd_stats.index[-1]])<1e-6)))
            and self.pd_stats.count(0)[0]> (self.skip_first_n + self.lastₙ)):

                intg.reset_opt_stats = True
                intg.bad_sim = False

                intg.opt_cost_mean = self.pd_stats['cost'].tail(self.lastₙ).mean()
                intg.opt_cost_std  = self.pd_stats['cost'].tail(self.lastₙ).std()
        return

    def bcast_status(self, intg):
        intg.reset_opt_stats = self.comm.bcast(intg.reset_opt_stats, root=0)
        intg.bad_sim         = self.comm.bcast(intg.bad_sim,         root=0)
        