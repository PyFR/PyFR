# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin


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
        comm, rank, root = get_comm_rank_root()
        if rank == root:
            print('n eddies = {}'.format(self.N))

        # Initialize the previous time to the current one.
        self.tprev = intg.tcurr

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Input Reynolds stress
        self.reystress = np.array(self.cfg.getliteral(cfgsect, 'ReynoldsStress'))
        # Reystress = np.array([[R_11, R_21, R_31],
        #               [R_21, R_22, R_32],
        #               [R_31, R_32, R_33]])

        # characteristic lengths,3x3 matrix (XYZ x UVW). This corresponds to
        # the radii of influence of the synthetic eddies, i.e. to the one-side
        # integral length scale.
        self.lturb = np.array(self.cfg.getliteral(cfgsect, 'lturb'))

        # Bulk velocity
        self.Ubulk = self.cfg.getfloat(cfgsect, 'Ubulk')

        # bulk velocity direction, either 0,1, or 2 (i.e. x,y,z)
        # this is also the normal to the synthetic inflow plane
        self.Ubulkdir = self.cfg.getint(cfgsect, 'Ubulk-dir')

        # # Frozen turbulence hypothesis to get characteristic times in each vel.
        # # component.
        # self.tturb = self.lturb[self.Ubulkdir,:]/self.Ubulk

        # update frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps', 1)

        # Center point of the synthetic inflow plane.
        self.ctr = np.array(self.cfg.getliteral(cfgsect, 'center'))

        # Determine the dimensions of the box of eddies and store the extension
        # in the streamwise direction.
        inflow = np.array(self.cfg.getliteral(cfgsect, 'plane-dimensions'))

        lstreamwise = 2.0*np.max(self.lturb[self.Ubulkdir])
        self.box_xmax = self.ctr[self.Ubulkdir] + lstreamwise/2.0
        self.box_xmin = self.ctr[self.Ubulkdir] - lstreamwise/2.0

        dirs = [i for i in range(self.ndims) if i != self.Ubulkdir]

        self.box_dims = np.zeros(self.ndims)
        self.box_dims[self.Ubulkdir] = lstreamwise
        self.box_dims[dirs] = inflow

        # Allocate the memory for the working arrays
        self.eddies_loc = np.empty((self.ndims, self.N))
        self.eddies_strength = np.empty((self.ndims, self.N))

        # Populate the box of eddies
        self.create_eddies(intg.tcurr)

        # Update the backend
        self.update_backend()

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

    def __call__(self, intg):
        # Return if not active or no action is due
        if intg.nacptsteps % self.nsteps:
            return

        t = intg.tcurr

        # Update the streamwise location of the eddies and check whether any are
        # outside of the box. If so generate new ones.
        self.eddies_loc[self.Ubulkdir] += (t - self.tprev)*self.Ubulk
        neweddies = self.eddies_loc[self.Ubulkdir] > self.box_xmax

        if neweddies.any():
            self.create_eddies(t, neweddies)

        # Update the backend
        self.update_backend()

        # Update the previous time
        self.tprev = t
