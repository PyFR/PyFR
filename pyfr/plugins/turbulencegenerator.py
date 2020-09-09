# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
import os

from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin
from pyfr.nputil import npeval

def get_lref_etype(cfg, cfgsect, constants, loc, ndims):
    # Bring simulation constants into scope
    vars = constants

    # get the physical location. loc should have a shape like (ndims, -1)
    vars.update(dict(zip('xyz', loc)))

    # maximum component-wise
    lturbmax = [[np.max(npeval(cfg.getexpr(cfgsect, f'l{i}{j}'), vars))
                 for j in range(ndims)] for i in range(ndims)]

    #lref = max_j (l_ij)
    return np.max(np.array(lturbmax), axis=1)

def eval_expr(expr, constants, loc):
    # Bring simulation constants into scope
    vars = constants

    # get the physical location
    vars.update(dict(zip('xyz', loc)))

    # evaluate the expression at the given location
    return npeval(expr, vars)

def get_lturbref(cfg, cfgsect, constants, ndims):
    # try to compute them if not specified by the user
    if cfg.hasopt('soln-plugin-turbulencegenerator', 'lturbref'):
        return np.array(cfg.getliteral(cfgsect, 'lturbref'))
    else:
        try:
            # this works only if all lenghts are constant and not space dependent.
            vars = constants
            lturb = [[npeval(cfg.getexpr(cfgsect, f'l{i}{j}'), vars)
                      for j in range(ndims)] for i in range(ndims)]
            return np.max(np.array(lturb), axis=1)
        except:
            msg = 'Could not determine lturbref automatically. Set it in the .ini file.'
            raise RuntimeError(msg)


class TurbulenceGeneratorPlugin(BasePlugin):
    name = 'turbulencegenerator'
    systems = ['ac-navier-stokes', 'navier-stokes']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        # Set the pointer to the elements
        self.elemap = intg.system.ele_map

        # Number of eddies. TODO ugly!
        for ele in self.elemap.values():
            self.N = ele.N

        # MPI info
        comm, self.rank, self.root = get_comm_rank_root()
        if self.rank == self.root:
            print('n eddies = {}'.format(self.N))

        # Initialize the previous time to the current one.
        self.tprev = intg.tcurr

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # characteristic lengths,3x3 matrix (XYZ x UVW). This corresponds to
        # the radii of influence of the synthetic eddies, i.e. to the one-side
        # integral length scale.
        # self.lturb = np.array(self.cfg.getliteral(cfgsect, 'lturb'))

        # reference turbulent = np.max(lturb, axis=1)
        # useful to have it in the .ini file to avoid problems with weird domains
        self.lturbref = get_lturbref(self.cfg, cfgsect, self._constants, self.ndims)
        # # or
        # # physical location of solution points.
        # plocs = []
        # for ele in self.elemap.values():
        #     pname = 'qpts' if 'div-flux' in ele.antialias else 'upts'
        #     plocs.append(ele.ploc_at_np(pname).swapaxes(1, 0).reshape(self.ndims, -1))
        # self.lturbref = self.get_lref(cfgsect, plocs)
        # print('lref = ' + f'{self.lref}')

        # Bulk velocity
        self.Ubulk = self.cfg.getfloat(cfgsect, 'Ubulk')

        # bulk velocity direction, either 0,1, or 2 (i.e. x,y,z)
        # this is also the normal to the synthetic inflow plane
        self.Ubulkdir = self.cfg.getint(cfgsect, 'Ubulk-dir')

        # update frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps', 1)

        # Center point of the synthetic inflow plane.
        self.ctr = np.array(self.cfg.getliteral(cfgsect, 'center'))

        # Determine the dimensions of the box of eddies and store the extension
        # in the streamwise direction.
        inflow = np.array(self.cfg.getliteral(cfgsect, 'plane-dimensions'))

        lstreamwise = 2.0*self.lturbref[self.Ubulkdir]
        self.box_xmax = self.ctr[self.Ubulkdir] + lstreamwise/2.0
        self.box_xmin = self.ctr[self.Ubulkdir] - lstreamwise/2.0

        dirs = [i for i in range(self.ndims) if i != self.Ubulkdir]

        self.box_dims = np.zeros(self.ndims)
        self.box_dims[self.Ubulkdir] = lstreamwise
        self.box_dims[dirs] = inflow

        # Allocate the memory for the working arrays
        self.eddies_loc = np.empty((self.ndims, self.N))
        self.eddies_strength = np.empty((self.ndims, self.N))

        # file to read/write the eddies' properties
        self.basedir = self.cfg.getpath(cfgsect, 'basedir', './')
        self.basename = self.cfg.get(cfgsect, 'basename', 'eddies-{t}.csv')

        # Append the relevant extension
        if not self.basename.endswith('.csv'):
            self.basename += '.csv'

        # Output time step and last output time of the eddies file.
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # Populate the box of eddies
        if os.path.isfile(self._get_output_path(intg.tcurr)) and intg.isrestart:
            data = np.loadtxt(self._get_output_path(intg.tcurr), delimiter=',')
            self.eddies_loc = data[:, :self.ndims].swapaxes(0,1)
            self.eddies_strength = data[:, self.ndims:].swapaxes(0,1)
        else:
            self.create_eddies(intg.tcurr)

        # Update the backend
        self.update_backend()

        # If we're not restarting then call ourself to write out the initial eddies
        if not intg.isrestart:
            self.tout_last -= self.dt_out
            self(intg)

    def get_lref(self, cfgsect, locs):
        # loop over the locations of all element types
        lref = np.zeros(self.ndims)
        for loc in locs:
            lref_etype = get_lref_etype(self.cfg, cfgsect, self._constants, loc,
                                        self.ndims)
            lref = np.maximum(lref, lref_etype)

        # get the max across all ranks.
        #WARNING this works only if the expression in the .ini file is formulated
        # such that it gives the correct values anywhere in the domain
        # so you might want to use a max,min function in there
        lref = comm.allreduce(lref, op=get_mpi('max'))
        print('lref = ' + f'{lref}')
        return lref

    @staticmethod
    def random_seed(t):
        return int(t*10000) + 23

    def create_eddies(self, t, neweddies=None):
        # Eddies to be generated: number and indices.
        N    = self.N           if neweddies is None else np.count_nonzero(neweddies)
        idxe = np.full(N, True) if neweddies is None else neweddies

        # Generate the new random locations and strengths. Set the seed so all
        # processors generate the same eddies.
        np.random.seed(self.random_seed(t))
        random = np.random.uniform(-1, 1, size=(self.ndims*2, N))

        # Make sure the strengths are only +/- ones.
        strengths = random[self.ndims:]
        pos = strengths >= 0.0
        neg = strengths <  0.0
        strengths[pos] = np.ceil(strengths[pos])
        strengths[neg] = np.floor(strengths[neg])

        self.eddies_strength[:, idxe] = strengths

        # Locations.
        loc = random[:self.ndims]
        self.eddies_loc[:, idxe] = loc*(self.box_dims[:,np.newaxis]/2.0) + self.ctr[:,np.newaxis]

        # New eddies are generated at the inlet of the box, except at startup,
        # when the variable neweddies in None.
        if neweddies is not None:
            self.eddies_loc[self.Ubulkdir, idxe] = self.box_xmin

    def update_backend(self):
        for ele in self.elemap.values():
            ele.eddies_loc.set(self.eddies_loc)

            ele.eddies_strength.set(self.eddies_strength)

    def _get_output_path(self, tcurr):
        # Substitute {t} for the current time
        fname = self.basename.format(t=tcurr)

        return os.path.join(self.basedir, fname)

    def __call__(self, intg):
        tdiff = intg.tcurr - self.tout_last
        dowrite = tdiff >= self.dt_out - self.tol
        doupdate = intg.nacptsteps % self.nsteps == 0

        # Return if not active or no action is due
        if dowrite or doupdate:
            # Update the streamwise location of the eddies and check whether any
            # are outside of the box. If so, generate new ones.
            self.eddies_loc[self.Ubulkdir] += (intg.tcurr - self.tprev)*self.Ubulk
            neweddies = self.eddies_loc[self.Ubulkdir] > self.box_xmax

            if neweddies.any():
                self.create_eddies(intg.tcurr, neweddies)

            # Update the backend
            self.update_backend()

            # Update the previous time an update was done.
            self.tprev = intg.tcurr

            # write to file if requested.
            if dowrite:
                if self.rank == self.root:
                    data = np.hstack((self.eddies_loc.swapaxes(0,1),
                                      self.eddies_strength.swapaxes(0,1)))
                    np.savetxt(self._get_output_path(intg.tcurr), data,
                               delimiter=',')

                # Update the last output time
                self.tout_last = intg.tcurr
