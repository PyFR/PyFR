# -*- coding: utf-8 -*-

import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.nputil import npeval
from pyfr.plugins.base import BasePlugin, init_csv
from pyfr.quadrules import get_quadrule


class IntegratePlugin(BasePlugin):
    name = 'integrate'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()
        
        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Expressions to integrate
        c = self.cfg.items_as('constants', float)
        self.exprs = [self.cfg.getexpr(cfgsect, k, subs=c)
                      for k in self.cfg.items(cfgsect)
                      if k.startswith('int-')]

        # Gradient pre-processing
        self._init_gradients(intg)

        # Save a reference to the physical solution point locations
        self.plocs = intg.system.ele_ploc_upts

        # Integration parameters
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        
        # The root rank needs to open the output file
        if rank == root:
            header = ['t'] + [k for k in self.cfg.items(cfgsect) if k.startswith('int-')]

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))
            
        self.elminfo = []
        # Iterate over the element types
        for ename, eles in intg.system.ele_map.items():
            # Locations of each solution point
            plocupts = eles.ploc_at_np('upts')
            
            # Jacobians
            jacs = 1/eles.rcpdjac_at_np('upts')
        
            # Weights
            rname = self.cfg.get('solver-elements-' + ename, 'soln-pts')
            wts = get_quadrule(ename, rname, eles.nupts).wts
            
            # Save
            self.elminfo.append((plocupts, jacs, wts))

    def _init_gradients(self, intg):
        # Determine what gradients, if any, are required
        self._gradpnames = gradpnames = set()
        for ex in self.exprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        # If gradients are required then form the relevant operators
        if gradpnames:
            self._gradop, self._rcpjact = [], []

            for eles in intg.system.ele_map.values():
                self._gradop.append(eles.basis.m4)

                # Get the smats at the solution points
                smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)

                # Get |J|^-1 at the solution points
                rcpdjac = eles.rcpdjac_at_np('upts')

                # Product to give J^-T at the solution points
                self._rcpjact.append(smat*rcpdjac)

    def _eval_exprs(self, intg):
        # Get the primitive variable names
        pnames = self.elementscls.privarmap[self.ndims]

        exprs = np.zeros(len(self.exprs))
        # Iterate over each element type in the simulation
        for i, (soln, (plocs, jacs, wts)) in enumerate(zip(intg.soln, self.elminfo)):
    
            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln.swapaxes(0, 1),
                                                 self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns))
            subs.update(zip('xyz', plocs.swapaxes(0, 1)))

            # Compute any required gradients
            if self._gradpnames:
                # Gradient operator and J^-T matrix
                gradop, rcpjact = self._gradop[i], self._rcpjact[i]
                nupts = gradop.shape[1]

                for pname in self._gradpnames:
                    psoln = subs[pname]

                    # Compute the transformed gradient
                    tgradpn = gradop @ psoln
                    tgradpn = tgradpn.reshape(self.ndims, nupts, -1)

                    # Untransform this to get the physical gradient
                    gradpn = np.einsum('ijkl,jkl->ikl', rcpjact, tgradpn)
                    gradpn = gradpn.reshape(self.ndims, nupts, -1)
                    
                    for dim, grad in zip('xyz', gradpn):
                        subs[f'grad_{pname}_{dim}'] = grad

            for j,v in enumerate(self.exprs):
                # Accumulate integrated evaluated expressions
                exprs[j] += np.sum(wts[:, None]*jacs*npeval(v, subs))
            
        # Stack up the expressions for each element type and return
        return exprs

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            # MPI info
            comm, rank, root = get_comm_rank_root()
            
            # Evaluate the integation expressions
            iintex = self._eval_exprs(intg)

            # Reduce and output if we're the root rank
            if rank != root:
                comm.Reduce(iintex, None, op=get_mpi('sum'), root=root)
            else:
                comm.Reduce(get_mpi('in_place'), iintex, op=get_mpi('sum'), root=root)

                # Build the row
                row = [intg.tcurr] + iintex.tolist()

                # Write
                print(','.join(str(r) for r in row), file=self.outf)

                # Flush to disk
                self.outf.flush()
                
