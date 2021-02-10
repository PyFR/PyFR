# -*- coding: utf-8 -*-

import re

import numpy as np

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

        # Underlying system
        system = intg.system

        # Underlying system elements class
        self.elementscls = system.elementscls

        # Expressions to integrate
        c = self.cfg.items_as('constants', float)
        self.exprs = [self.cfg.getexpr(cfgsect, k, subs=c)
                      for k in self.cfg.items(cfgsect)
                      if k.startswith('int-')]

        # Integration region pre-processing
        rinfo = self._prepare_region_info(intg)

        # Gradient pre-processing
        self._init_gradients(intg, rinfo)

        # Save a reference to the physical solution point locations
        self.plocs = system.ele_ploc_upts

        # Integration parameters
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # The root rank needs to open the output file
        if rank == root:
            header = ['t'] + [k for k in self.cfg.items(cfgsect)
                              if k.startswith('int-')]

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

        # Prepare the per element-type info list
        self.eleinfo = []
        for (ename, eles), (eset, emask) in zip(system.ele_map.items(), rinfo):
            # Locations of each solution point
            ploc = eles.ploc_at_np('upts')[..., eset]
            ploc = ploc.swapaxes(0, 1)

            # Jacobian determinants
            rcpdjacs = eles.rcpdjac_at_np('upts')[:, eset]

            # Quadature weights
            rname = self.cfg.get(f'solver-elements-{ename}', 'soln-pts')
            wts = get_quadrule(ename, rname, eles.nupts).wts

            # Save
            self.eleinfo.append((ploc, wts[:, None] / rcpdjacs, eset, emask))

    def _prepare_region_info(self, intg):
        # All elements
        if self.cfg.get(self.cfgsect, 'region', '*') == '*':
            return [(slice(None), ([], []))]*len(intg.system.ele_types)
        # Elements inside of a box
        else:
            x0, x1 = self.cfg.getliteral(self.cfgsect, 'region')

            rinfo = []
            for etype in intg.system.ele_types:
                pts = intg.system.mesh[f'spt_{etype}_p{intg.rallocs.prank}']
                pts = np.moveaxis(pts, 2, 0)

                # Determine which points are inside the box
                inside = np.ones(pts.shape[1:], dtype=np.bool)
                for l, p, u in zip(x0, pts, x1):
                    inside &= (l <= p) & (p <= u)

                if np.all(inside):
                    rinfo.append((slice(None), ([], [])))
                else:
                    # Determine which elements have some points inside the box
                    eset = np.any(inside, axis=0).nonzero()[0]

                    # Mask any points outside of the box
                    emask = (~inside[:, eset]).nonzero()

                    rinfo.append((eset, emask))

            return rinfo

    def _init_gradients(self, intg, rinfo):
        # Determine what gradients, if any, are required
        gradpnames = set()
        for ex in self.exprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        privarmap = self.elementscls.privarmap[self.ndims]
        self._gradpinfo = [(pname, privarmap.index(pname))
                            for pname in gradpnames]

    def _eval_exprs(self, intg):
        intvals = np.zeros(len(self.exprs))

        # Get the primitive variable names
        pnames = self.elementscls.privarmap[self.ndims]

        # Compute primitive gradients if required
        if self._gradpinfo:
            grads_eles = self._grad_pvars(intg)

        # Iterate over each element type in the simulation
        for i, (soln, eleinfo) in enumerate(zip(intg.soln, self.eleinfo)):
            plocs, wts, eset, emask = eleinfo

            # Subset and transpose the solution
            soln = soln[..., eset].swapaxes(0, 1)

            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln, self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns))
            subs.update(zip('xyz', plocs))

            # Prepare any required gradients
            for pname, idx in self._gradpinfo:
                for dim, grad in zip('xyz', grads_eles[i][idx]):
                    subs[f'grad_{pname}_{dim}'] = grad

            for j, v in enumerate(self.exprs):
                # Evaluate the expression at each point
                iex = wts*npeval(v, subs)

                # Accumulate
                intvals[j] += np.sum(iex) - np.sum(iex[emask])

        return intvals

    def _grad_pvars(self, intg):
        grads_eles = []

        # Iterate over each element type in the simulation
        for i, (soln, eleinfo) in enumerate(zip(intg.soln, self.eleinfo)):
            eset, emask = eleinfo[2:]

            # Subset and transpose the solution
            soln = soln[..., eset].swapaxes(0, 1)

            # Rearrange and subset gradient data
            grad_soln = np.rollaxis(intg.grad_soln[i], 2)[..., eset]

            # Transform from conservative to primitive gradients
            pgrads = self.elementscls.grad_con_to_pri(soln, grad_soln, self.cfg)

            # Store the gradients
            grads_eles.append(pgrads)

        return grads_eles

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
                comm.Reduce(get_mpi('in_place'), iintex, op=get_mpi('sum'),
                            root=root)

                # Write
                print(intg.tcurr, *iintex, sep=',', file=self.outf)

                # Flush to disk
                self.outf.flush()
