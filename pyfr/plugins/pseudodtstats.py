from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BasePlugin, init_csv

import numpy as np

class PseudodtStatsPlugin(BasePlugin):
    name = 'pseudodt_stats'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect:str, prefix):
        super().__init__(intg, cfgsect, prefix)


        self.stats = []

        self.tₚ = intg.tcurr

        self.fvars   = intg.system.elementscls.convarmap[self.ndims]
        self.e_types = intg.system.ele_types

        # Maximum of 3 Levels of abstraction for the stats of pseudo-dt field
        self.dtau_stats = { 'n' : {'all':0},
                         'res': {'all':0}|{p:{'all':0}                                     
                                           for p in self.fvars}, 
 
                         'min': {'all':0}|{p:{'all':0}|{e:{'all':0} 
                                                        for e in self.e_types} 
                                           for p in self.fvars}, 
 
                         'max': {'all':0}|{p:{'all':0}|{e:{'all':0} 
                                                        for e in self.e_types} 
                                           for p in self.fvars},
                        }

        self.abstraction = self.cfg.getint(self.cfgsect, 'abstraction', 0)

        if self.abstraction > 2:
            raise ValueError('abstraction > 2 has not been implemented yet')

        if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
            self.level = self.cfg.getint(self.cfgsect, 
                'level', intg.cfg.getint('solver','order'))

        csv_header =  'pseudo-steps, tcurr'
        for k, v in self.dtau_stats.items():
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
        if (self.rank == self.root) and (self.abstraction != 0):
            self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 500)
            self.outf = init_csv(self.cfg, cfgsect, csv_header)
        else:
            self.outf = None

        self.stored_tₚ = None
        self.last_appendable = None

    def __call__(self, intg):
        # Process the sequence of pseudo-residuals

        for (npiter, iternr, resid) in intg.pseudostepinfo:

            if iternr == 1:           # We can store the last step's data
                if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
                    Δτ_mats = intg.pseudointegrator.pintgs[self.level].dtau_mats
                else:
                    Δτ_mats = intg.pseudointegrator.dtau_mats
                
                if self.last_appendable != None:
                    if self.stored_tₚ != self.tₚ:
                        self.stats.append((f for f in self.last_appendable))
                        self.prev_npiter = npiter - 1
                else:
                    self.prev_npiter = 0 

                self.stored_tₚ = self.tₚ

            self.dtau_stats['n']['all'] = npiter-self.prev_npiter

            self.Δτ_statistics(intg, Δτ_mats)
            self.residual_statistics(intg, resid)
            self.last_appendable = (npiter, intg.tcurr, 
                                    *self.Δτ_stats_as_list(self.dtau_stats))

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

    def Δτ_statistics(self, intg, Δτ_mats):
        '''
            self.Δτ_mats is a list of matrices, one for each element type.
            Each matrix is a 3D array of shape (nupts, nvars, neles)
        '''

        for j, var in enumerate(self.fvars):
            for i, e_type in enumerate(self.e_types):

                # each element type, each soln point in element, each variable in (p, u, v, w)
                # Stats obtained over all elements
                self.dtau_stats['min'][var][e_type]['each'] = (
                                                    Δτ_mats[i][:, j, :].min(1))
                self.dtau_stats['max'][var][e_type]['each'] = (
                                                    Δτ_mats[i][:, j, :].max(1))

                # each element type, each variable in (p, u, v, w)
                # Stats obtained over all elements and element soln points

                self.dtau_stats['min'][var][e_type]['all'] = (
                                self.dtau_stats['min'][var][e_type]['each'].min())
                self.dtau_stats['max'][var][e_type]['all'] = (
                                self.dtau_stats['max'][var][e_type]['each'].max())

            # each variable in (p, u, v, w)
            # Stats obtained over all element types, elements and element soln points
            self.dtau_stats['min'][var]['all'] = min(
                [self.dtau_stats['min'][var][e_type]['all'] 
                                                    for e_type in self.e_types])
            self.dtau_stats['max'][var]['all'] = max(
                [self.dtau_stats['max'][var][e_type]['all'] 
                                                    for e_type in self.e_types])

            tₘₐₓ = np.array(self.dtau_stats['min'][var]['all'])
            if self.rank != self.root:
                self.comm.Reduce(tₘₐₓ       , None , op=mpi.MIN, root=self.root)
            else:
                self.comm.Reduce(mpi.IN_PLACE, tₘₐₓ, op=mpi.MIN, root=self.root)
            self.dtau_stats['min'][var][e_type]['all'] = tₘₐₓ

            tₘᵢₙ = np.array(self.dtau_stats['max'][var]['all'])
            if self.rank != self.root:
                self.comm.Reduce(tₘᵢₙ       , None , op=mpi.MAX, root=self.root)
            else:
                self.comm.Reduce(mpi.IN_PLACE, tₘᵢₙ, op=mpi.MAX, root=self.root)
            self.dtau_stats['max'][var][e_type]['all'] = tₘᵢₙ

        # Stats obtained over 
        #   all element types, elements, variable 
        #   in (p, u, v, w) 
        #   and element soln points
        self.dtau_stats['min']['all'] = min([self.dtau_stats['min'][var]['all'] 
                                           for var in self.fvars])
        self.dtau_stats['max']['all'] = max([self.dtau_stats['max'][var]['all'] 
                                           for var in self.fvars])

    def residual_statistics(self, intg, resid):
        '''
            Use a list of numpy arrays, one for each element type.
            Each array is of shape(nvars,)
        '''
        resid = resid or (0,)*intg.system.nvars

        # each variable in (p, u, v, w)
        for j, var in enumerate(self.fvars):
            self.dtau_stats['res'][var]['all'] = resid[j]

        if self.cfg.get('solver-time-integrator','pseudo-resid-norm')=='uniform':
            self.dtau_stats['res']['all'] = max([self.dtau_stats['res'][var]['all'] 
                                           for var in self.fvars])
        elif self.cfg.get('solver-time-integrator', 'pseudo-resid-norm')=='l2':
            self.dtau_stats['res']['all'] = sum([self.dtau_stats['res'][var]['all'] 
                                           for var in self.fvars])
        elif self.cfg.get('solver-time-integrator', 'pseudo-resid-norm')=='l4':
            self.dtau_stats['res']['all'] = sum([self.dtau_stats['res'][var]['all'] 
                                           for var in self.fvars])
        elif self.cfg.get('solver-time-integrator', 'pseudo-resid-norm')=='l8':
            self.dtau_stats['res']['all'] = sum([self.dtau_stats['res'][var]['all'] 
                                           for var in self.fvars])
        else:
            raise ValueError('Unknown time integrator ')

    def residual_statistics_next_gen(self, intg, resid):
        '''
            Use a list of numpy arrays, one for each element type.
            Each array is of shape(nvars,)
        '''

        for j, var in enumerate(self.fvars):
            for i, e_type in enumerate(self.e_types):

                # each element type, each variable in (p, u, v, w)
                # Stats obtained over all elements
                self.dtau_stats['res'][var][e_type]['each'] = resid[i][j]

                # each element type, each variable in (p, u, v, w)
                # Stats obtained over all elements and element soln points

                res = np.array(self.dtau_stats['res'][var][e_type]['each'].max())

                if self.rank != self.root:
                    self.comm.Reduce(res, None , op=mpi.SUM, 
                                     root = self.root)
                else:
                    self.comm.Reduce(mpi.IN_PLACE, res, op=mpi.SUM, 
                                     root = self.root)

                self.dtau_stats['res'][var][e_type]['all'] = res

            # each variable in (p, u, v, w)
            # Stats obtained over all element types, elements and element soln points
            self.dtau_stats['res'][var]['all'] = max([self.dtau_stats['res'][var][e_type]['all'] 
                                                    for e_type in self.e_types])

        # Stats obtained 
        #   over all element types, elements, variable 
        #   in (p, u, v, w) 
        #   and element soln points
        self.dtau_stats['res']['all'] = max([self.dtau_stats['res'][var]['all'] 
                                           for var in self.fvars])

    def Δτ_stats_as_list(self, Δτ_stats):
        Δτ_stats_list = []
        for v in Δτ_stats.values():
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