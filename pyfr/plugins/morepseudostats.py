from __future__ import annotations

import os
import numpy as np
from sigfig import round

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin, init_csv

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyfr.integrators.dual.phys.base import BaseDualIntegrator

class MorePseudoStatsPlugin(BasePlugin):
    name = 'morepseudostats'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg:BaseDualIntegrator, cfgsect:str, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 500)

        self.count = 0
        self.stats:list[tuple[int,float,int,float,float,float,float,float]] = []


        self.__dt_decimals = str(intg._dt)[::-1].find('.')+1

        self.tprev = round(intg.tcurr, decimals=self.__dt_decimals)

        fvars = ','.join(intg.system.elementscls.convarmap[self.ndims])
        num_vars =   len(intg.system.elementscls.convarmap[self.ndims])
        # MPI info
        comm, self.rank, self.root = get_comm_rank_root()

        self.last_appendable = (0, intg.tcurr, 0,) \
                                + (0, 0,) \
                                + (num_vars*(0,))
        self.order = intg.cfg.getint('solver','order')

        # The root rank needs to open the output file
        if self.rank == self.root:
            self.outf = init_csv(self.cfg, cfgsect, 'n,t,i,max,min,' + fvars)
            self.n_storage = []
        else:
            self.outf = None

        self.last_tprev = None
        self.last_stored_iternr = 0

    def __call__(self, intg:BaseDualIntegrator):
        # Process the sequence of pseudo-residuals

        for (npiter, iternr, resid) in intg.pseudostepinfo:
            resid = resid or ('-',)*intg.system.nvars

            if iternr == 1:           # We can store the last step's data
                psdt = tuple(self.pseudo_dt_stats(intg).values())
                    
                if self.last_appendable[0] !=0 and self.last_tprev != self.tprev:
                    self.stats.append((f for f in self.last_appendable))
                    self.last_stored_iternr = self.last_appendable[0]

                self.last_tprev = self.tprev
            self.last_appendable = (npiter, self.tprev, self.last_appendable[0] - self.last_stored_iternr+1,)  + psdt + resid

        # Update the total step count and save the current time
        self.count += len(intg.pseudostepinfo)
        
        
        l_dt    = str(intg._dt  )[::-1].find('.')+1
            
        if self.__dt_decimals<l_dt:
            tc = intg.tcurr
        else:            
            tc = round(intg.tcurr, decimals=self.__dt_decimals)
        
        self.tprev = tc

        # If we're the root rank then output
        if self.outf:
            for s in self.stats:
                print(*s, sep=f',', file=self.outf)

            # Periodically flush to disk
            if intg.nacptsteps % self.flushsteps == 0:
                self.outf.flush()

            #self.n_storage.append([self.tprev, iternr])

        # Reset the stats
        self.stats = []

    def pseudo_dt_stats(self, intg:BaseDualIntegrator) -> dict:    
        """
            ## Collect pseudo-dt statistics within each physical step for all elements. 
            ### mean, standard deviation, minimum, and maximum values are currently collected.
            The data may be used to later fix the value of pseudo-dt when `pseudo-controller = None`.
            
        """        
        ##if self.rank != self.root:  raise Exception("Do this in the root node.")

        if 'solver-dual-time-integrator-multip' in intg.cfg.sections():

            return intg.pseudointegrator.pintg.dtau_stats
            
#            return {'maxes':[intg.pseudointegrator.pintgs[i].dtau_stats['min'] for i in range(5)]}

        else:
            return intg.pseudointegrator.dtau_stats
       
    def init_csv(self, cfgsect, header_string):
        
        self.fname = self.cfg.get(cfgsect, 'file')
        if not self.fname.endswith('.csv'):
            self.fname += '.csv'
        with open(self.fname, 'a') as outf:
            if os.path.getsize(self.fname) == 0 and self.cfg.getbool(cfgsect, header_string, True):
                print(header_string, file=outf)
                
    def root_check(self):
        if self.rank != self.root:  raise Exception("Do this in the root node.")

    @staticmethod
    def print7(nn:list) -> list:
        ss = []
        for n in nn:
            ss.append( int(n) if n%1==0 else float("{0:.7f}".format(n)) )
        return ss