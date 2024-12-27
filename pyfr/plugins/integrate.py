import re

import numpy as np

from pyfr.inifile import NoOptionError
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import npeval
from pyfr.plugins.base import BaseSolnPlugin, init_csv
from pyfr.quadrules import get_quadrule
from pyfr.regions import ConstructiveRegion
from pyfr.util import first


class IntegratePlugin(BaseSolnPlugin):
    name = 'integrate'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Underlying system
        system = intg.system

        # Underlying system elements class
        self.elementscls = system.elementscls

        # Primitive variables
        self.privars = first(intg.system.ele_map.values()).privars

        # Expressions to integrate
        c = self.cfg.items_as('constants', float)
        self.exprs = [self.cfg.getexpr(cfgsect, k, subs=c)
                      for k in self.cfg.items(cfgsect, prefix='int-')]

        # Add optional integral L-p degree
        if self.cfg.get(cfgsect, 'norm', 'none').lower() != 'none':
            self.lp = self.cfg.getfloat(cfgsect, 'norm')

            # Modify expressions for L-p norm
            if self.lp == 1 or self.lp == float('inf'):
                self.exprs = [f'abs({e})' for e in self.exprs]
                self._post_func = lambda x: x
            else:
                self.exprs = [f'pow(abs({e}), {self.lp})' for e in self.exprs]
                self._post_func = lambda x: x**(1 / self.lp)
        else:
            self.lp = None
            self._post_func = lambda x: x

        # Set MPI reduction op
        self.mpi_op = mpi.MAX if self.lp == float('inf') else mpi.SUM

        # Integration region pre-processing
        esetmask = self._prepare_esetmask(intg)

        # Gradient pre-processing
        self._init_gradients()

        # Save a reference to the physical solution point locations
        self.plocs = system.ele_ploc_upts

        # Integration parameters
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # The root rank needs to open the output file
        if rank == root:
            header = ['t', *self.cfg.items(cfgsect, prefix='int-')]

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

        # Prepare the per element-type info list
        self.eleinfo = eleinfo = []
        for ename, eles in system.ele_map.items():
            # Obtain quadrature info
            rname = self.cfg.get(f'solver-elements-{ename}', 'soln-pts')

            try:
                # Quadrature rule (default to that of the solution points)
                qrule = self.cfg.get(cfgsect, f'quad-pts-{ename}', rname)

                # Quadrature rule degree
                try:
                    qdeg = self.cfg.getint(cfgsect, f'quad-deg-{ename}')
                except NoOptionError:
                    qdeg = self.cfg.getint(cfgsect, 'quad-deg')

                r = get_quadrule(ename, qrule, qdeg=qdeg)

                # Interpolation to quadrature points matrix
                m0 = eles.basis.ubasis.nodal_basis_at(r.pts)
            except NoOptionError:
                # Default to the quadrature rule of the solution points
                r = get_quadrule(ename, rname, eles.nupts)
                m0 = None

            # Locations of each quadrature point
            ploc = eles.ploc_at_np(r.pts).swapaxes(0, 1)

            # Obtain the region mask
            eset, emask = esetmask(ploc)

            # Use this to subset the quadrature points
            ploc = ploc[..., eset]

            # Jacobian determinants at each quadrature point
            rcpdjacs = eles.rcpdjac_at_np(r.pts)[:, eset]

            # Save
            eleinfo.append((ploc, r.wts[:, None] / rcpdjacs, m0, eset, emask))

    def _prepare_esetmask(self, intg):
        region = self.cfg.get(self.cfgsect, 'region', '*')

        # All elements
        if region == '*':
            return lambda pts: (slice(None), ([], []))
        # Elements inside of a paramaterised shape
        else:
            crgn = ConstructiveRegion(region)

            def esetmask(pts):
                inside = crgn.pts_in_region(np.moveaxis(pts, 0, 2))

                if np.all(inside):
                    return slice(None), ([], [])
                else:
                    # Determine which elements have some points inside the box
                    eset = np.any(inside, axis=0).nonzero()[0]

                    # Mask any points outside of the box
                    emask = (~inside[:, eset]).nonzero()

                    return eset, emask

            return esetmask

    def _init_gradients(self):
        # Determine what gradients, if any, are required
        gradpnames = set()
        for ex in self.exprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        self._gradpinfo = [(pname, self.privars.index(pname))
                           for pname in gradpnames]

    def _eval_exprs(self, intg):
        intvals = np.zeros(len(self.exprs))

        # Compute the gradients
        if self._gradpinfo:
            grad_soln = intg.grad_soln

        # Iterate over each element type in the simulation
        for i, (soln, eleinfo) in enumerate(zip(intg.soln, self.eleinfo)):
            plocs, wts, m0, eset, emask = eleinfo

            # Subset and transpose the solution
            soln = soln[..., eset].swapaxes(0, 1)

            # Interpolate the solution to the quadrature points
            if m0 is not None:
                soln = m0 @ soln

            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln, self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(self.privars, psolns), t=intg.tcurr)
            subs |= dict(zip('xyz', plocs))

            # Prepare any required gradients
            if self._gradpinfo:
                grads = np.rollaxis(grad_soln[i], 2)[..., eset]

                # Interpolate the gradients to the quadrature points
                if m0 is not None:
                    grads = m0 @ grads

                # Transform from conservative to primitive gradients
                pgrads = self.elementscls.grad_con_to_pri(soln, grads,
                                                          self.cfg)

                # Add them to the substitutions dictionary
                for pname, idx in self._gradpinfo:
                    for dim, grad in zip('xyz', pgrads[idx]):
                        subs[f'grad_{pname}_{dim}'] = grad

            for j, v in enumerate(self.exprs):
                # Evaluate the expression at each point
                e = npeval(v, subs)

                # Either max or quadrature and sum
                if self.lp == float('inf'):
                    intvals[j] = max(intvals[j], np.amax(e))
                else:
                    iex = wts*e

                    # Accumulate
                    intvals[j] += np.sum(iex) - np.sum(iex[emask])

        return intvals

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            # MPI info
            comm, rank, root = get_comm_rank_root()

            # Evaluate the integation expressions
            iintex = self._eval_exprs(intg)

            # Reduce and output if we're the root rank
            if rank != root:
                comm.Reduce(iintex, None, op=self.mpi_op, root=root)
            else:
                comm.Reduce(mpi.IN_PLACE, iintex, op=self.mpi_op, root=root)

                # Apply post integration function
                iintex = (self._post_func(i) for i in iintex)

                # Write
                print(intg.tcurr, *iintex, sep=',', file=self.outf)

                # Flush to disk
                self.outf.flush()
