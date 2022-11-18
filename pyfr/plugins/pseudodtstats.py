# -*- coding: utf-8 -*-

from __future__ import annotations

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BasePlugin, init_csv

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyfr.integrators.dual.phys.base import BaseDualIntegrator

class PseudodtStatsPlugin(BasePlugin):
    name = 'pseudodt_stats'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg:BaseDualIntegrator, cfgsect:str, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 500)

        self.stats = []

        self.tₚ = intg.tcurr

        self.fvars   = intg.system.elementscls.convarmap[self.ndims]
        self.e_types = intg.system.ele_types

        # Maximum of 3 Levels of abstraction for the stats of pseudo-dt field
        self.Δτ_stats = {'max': {'all':0}|{p:{'all':0}|{e:{'all':0} for e in self.e_types} for p in self.fvars}, 
                           'min': {'all':0}|{p:{'all':0}|{e:{'all':0} for e in self.e_types} for p in self.fvars}}

        self.abstraction = self.cfg.getint(self.cfgsect, 'abstraction', 1)
        if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
            self.level = self.cfg.getint(self.cfgsect, 'level', intg.cfg.getint('solver','order'))

        csv_header =  'n,t,i'
        for k, v in self.Δτ_stats.items():
            if k != 'all':
                csv_header += f',{k}'
                for vk, vv in v.items():
                    if vk != 'all' and self.abstraction>1:
                        csv_header += f',{k}_{vk}'
                        for vvk in vv.keys():
                            if vvk != 'all' and self.abstraction>2:
                                csv_header += f',{k}_{vk}_{vvk}'

        # MPI info
        self.comm, self.rank, self.root = get_comm_rank_root()

        # The root rank needs to open the output file
        if self.rank == self.root:
            self.outf = init_csv(self.cfg, cfgsect, csv_header)
        else:
            self.outf = None

        self.last_tprev = None
        self.last_appendable = None

    def __call__(self, intg:BaseDualIntegrator):
        # Process the sequence of pseudo-residuals

        for (npiter, iternr, _) in intg.pseudostepinfo:

            if iternr == 1:           # We can store the last step's data
                if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
                    Δτ_mats = intg.pseudointegrator.pintgs[self.level].pseudodt_mats
                else:
                    Δτ_mats = intg.pseudointegrator.pseudodt_mats

                stats = self.Δτ_statistics(Δτ_mats)

                if self.last_appendable != None:
                    if self.last_tprev != self.tₚ:
                        self.stats.append((f for f in self.last_appendable))
                        self.prev_npiter = npiter - 1
                else:
                    self.prev_npiter = 0 

                self.last_tprev = self.tₚ

            self.last_appendable = (npiter, intg.tcurr, npiter-self.prev_npiter , *self.stats_as_list(stats))

        self.tₚ = intg.tcurr

        # If we're the root rank then output
        if self.outf:
            for s in self.stats:
                print(*s, sep=f',', file=self.outf)

            # Periodically flush to disk
            if intg.nacptsteps % self.flushsteps == 0:
                self.outf.flush()

        # Reset the stats
        self.stats = []

    def Δτ_statistics(self, Δτ_mats):
        '''
            self.pseudodt_mats is a list of matrices, one for each element type.
            Each matrix is a 3D array of shape (nupts, nvars, neles)
        '''

        for j, var in enumerate(self.fvars):
            for i, e_type in enumerate(self.e_types):

                # each element type, each soln point in element, each variable in (p, u, v, w)
                # Stats obtained over all elements
                self.Δτ_stats['max'][var][e_type]['each'] = Δτ_mats[i][:, j, :].max(1)
                self.Δτ_stats['min'][var][e_type]['each'] = Δτ_mats[i][:, j, :].min(1)

                # each element type, each variable in (p, u, v, w)
                # Stats obtained over all elements and element soln points

                tₘₐₓ = np.array(self.Δτ_stats['max'][var][e_type]['each'].max())
                tₘᵢₙ = np.array(self.Δτ_stats['min'][var][e_type]['each'].min())

                if self.rank != self.root:
                    self.comm.Reduce(tₘₐₓ       , None , op=mpi.MAX, root=self.root)
                else:
                    self.comm.Reduce(mpi.IN_PLACE, tₘₐₓ, op=mpi.MAX, root=self.root)

                if self.rank != self.root:
                    self.comm.Reduce(tₘᵢₙ       , None , op=mpi.MIN, root=self.root)
                else:
                    self.comm.Reduce(mpi.IN_PLACE, tₘᵢₙ, op=mpi.MIN, root=self.root)

                self.Δτ_stats['max'][var][e_type]['all'] = tₘₐₓ
                self.Δτ_stats['min'][var][e_type]['all'] = tₘᵢₙ

            # each variable in (p, u, v, w)
            # Stats obtained over all element types, elements and element soln points
            self.Δτ_stats['max'][var]['all'] = max([self.Δτ_stats['max'][var][e_type]['all'] for e_type in self.e_types])
            self.Δτ_stats['min'][var]['all'] = min([self.Δτ_stats['min'][var][e_type]['all'] for e_type in self.e_types])

        # Stats obtained over all element types, elements, variable in (p, u, v, w) and element soln points
        self.Δτ_stats['max']['all'] = max([self.Δτ_stats['max'][var]['all'] for var in self.fvars])
        self.Δτ_stats['min']['all'] = max([self.Δτ_stats['min'][var]['all'] for var in self.fvars])

        return self.Δτ_stats

    def stats_as_list(self, stats):
        Δτ_stats_list = []
        for v in stats.values():
            Δτ_stats_list.append(v['all'])               
            for vk, vv in v.items():
                if self.abstraction>1:                   
                    if isinstance(vv, dict):    
                        Δτ_stats_list.append(vv['all'])
                if vk != 'all':
                    for vvv in vv.values():
                        if self.abstraction>2:
                            if isinstance(vvv, dict):
                                Δτ_stats_list.append(vvv['all'])
        return Δτ_stats_list
