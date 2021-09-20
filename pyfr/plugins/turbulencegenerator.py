# -*- coding: utf-8 -*-

import numpy as np
import os

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin
from pyfr.nputil import npeval


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
            # this works only if all lenghts are constant and space independent
            vars = constants
            lturb = [[npeval(cfg.getexpr(cfgsect, f'l{i}{j}'), vars)
                      for j in range(ndims)] for i in range(ndims)]
            return np.max(np.array(lturb), axis=1)
        except:
            msg = ('Could not determine lturbref automatically. '
                   ' Set it in the .ini file.')
            raise RuntimeError(msg)


def computeNeddies(inflow, Ubulkdir, lturbref):
    ndims = lturbref.size
    dirs = [i for i in range(ndims) if i != Ubulkdir]

    inflowarea = np.prod(inflow)
    eddyarea = 4.0*np.prod(lturbref[dirs])  # 2 Ly x 2 Lz
    return int(inflowarea/eddyarea) + 1


class TurbulenceGeneratorPlugin(BasePlugin):
    name = 'turbulencegenerator'
    systems = ['ac-navier-stokes', 'navier-stokes']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix, eddies_loc=None, eddies_strength=None):
        super().__init__(intg, cfgsect, suffix)

        # do not allow multiple instances
        if suffix:
            raise RuntimeError('Only one instance of the turbulencegenerator '
               'plugin is allowed.')

        # Set the pointer to the elements
        self.elemap = intg.system.ele_map

        # MPI info
        _, self.rank, self.root = get_comm_rank_root()

        # Initialize the previous time to the current one.
        self.tprev = intg.tcurr

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # reference turbulent = np.max(lturb, axis=1). Useful to have it in
        # the .ini file to avoid problems with weird domains
        self.lturbref = get_lturbref(
            self.cfg, cfgsect, self._constants, self.ndims)

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

        # Number of eddies
        self.N = computeNeddies(inflow, self.Ubulkdir, self.lturbref)

        # Allocate the memory for the working arrays
        self.eddies_loc = np.empty((self.ndims, self.N))
        self.eddies_strength = np.empty((self.ndims, self.N))

        # Populate the box of eddies. If present, check that the eddies stored
        # in the previous solution are consistent with the current settings.
        restore = False if eddies_loc is None else True
        if intg.isrestart and restore:
            if eddies_loc.shape != (self.ndims, self.N):
                restore = False

        if restore:
            self.eddies_loc = eddies_loc
            self.eddies_strength = eddies_strength

            # update their location if necessary
            self._update_eddies(intg.tcurr)
        else:
            self._create_eddies(intg.tcurr)

        # Update the backend
        self._update_backend()

    @staticmethod
    def random_seed(t):
        return int(t*10000) + 23

    def serialise(self, intg):
        self._update_eddies(intg.tcurr)
        eddies = {'eddies_loc' : self.eddies_loc,
                  'eddies_strength' : self.eddies_strength}
        return eddies

    def _create_eddies(self, t, neweddies=None):
        # Eddies to be generated: number and indices.
        N = self.N if neweddies is None else np.count_nonzero(neweddies)
        idxe = np.full(N, True) if neweddies is None else neweddies

        # Generate the new random locations and strengths. Set the seed so all
        # processors generate the same eddies.
        np.random.seed(self.random_seed(t))
        random = np.random.uniform(-1, 1, size=(self.ndims*2, N))

        # Make sure the strengths are only +/- ones.
        strengths = random[self.ndims:]
        pos = strengths >= 0.0
        neg = strengths < 0.0
        strengths[pos] = np.ceil(strengths[pos])
        strengths[neg] = np.floor(strengths[neg])

        self.eddies_strength[:, idxe] = strengths

        # Locations.
        loc = random[:self.ndims]
        self.eddies_loc[:, idxe] = loc * \
            (self.box_dims[:, np.newaxis]/2.0) + self.ctr[:, np.newaxis]

        # New eddies are generated at the inlet of the box, except at startup,
        # when the variable neweddies in None.
        if neweddies is not None:
            self.eddies_loc[self.Ubulkdir, idxe] = self.box_xmin

    def _update_eddies(self, tcurr):
        # Update the streamwise location of the eddies and check whether
        # any are outside of the box. If so, generate new ones.
        self.eddies_loc[self.Ubulkdir] += (tcurr -
                                           self.tprev)*self.Ubulk
        neweddies = self.eddies_loc[self.Ubulkdir] > self.box_xmax

        if neweddies.any():
            self._create_eddies(tcurr, neweddies)

        # Update the previous time an update was done.
        self.tprev = tcurr

    def _update_backend(self):
        for ele in self.elemap.values():
            ele.eddies_loc.set(self.eddies_loc)

            ele.eddies_strength.set(self.eddies_strength)

    def __call__(self, intg):
        # Return if not active
        if intg.nacptsteps % self.nsteps == 0:
            # Update the location of the eddies
            self._update_eddies(intg.tcurr)

            # Update the backend
            self._update_backend()


