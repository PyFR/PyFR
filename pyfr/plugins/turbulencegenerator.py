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

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Input Reynolds stress
        self.reystress = np.array(self.cfg.getliteral(cfgsect, 'ReynoldsStress'))
        # Reystress = np.array([[R_11, R_21, R_31],
        #               [R_21, R_22, R_32],
        #               [R_31, R_32, R_33]])

        # characteristic lengths,3x3 matrix (XYZ x UVW). This is twice the radii
        # of influence of the synthetic eddies.
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

        lstreamwise = np.max(self.lturb[self.Ubulkdir])
        self.box_xmax = self.ctr[self.Ubulkdir] + lstreamwise/2.0

        dirs = [i for i in range(self.ndims) if i != self.Ubulkdir]

        self.box_dims = np.zeros(self.ndims)
        self.box_dims[self.Ubulkdir] = lstreamwise
        self.box_dims[dirs] = inflow

        # # Determine the number of eddies.
        # inflowarea = np.prod(inflow)
        # eddyarea = np.prod(self.lturb[dirs])
        # self.N = int(inflowarea/eddyarea) + 1

        # Allocate the memory for the working arrays
        self.eddies_time = np.empty((self.N))
        self.eddies_loc = np.empty((self.ndims, self.N))
        self.eddies_strength = np.empty((self.ndims, self.N))

        # Populate the box of eddies
        self.create_eddies(intg.tcurr)

    def create_eddies(self, t, neweddies=None):
        # Eddies to be generated: number and indices.
        N    = self.N           if neweddies is None else np.count_nonzero(neweddies)
        idxe = np.full(N, True) if neweddies is None else neweddies

        # Generation time.
        self.eddies_time[idxe] = t

        # Generate the new random locations and strengths. Set the seed so all
        # processors generate the same eddies.
        seed = int(t*1000) + 23
        np.random.seed(seed)
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
        self.eddies_loc[:, idxe] = loc*self.box_dims[:,np.newaxis]/2.0 + self.ctr[:,np.newaxis]

        # Update the backend
        for ele in self.elemap.values():
            # Broadcast the arrays to fit the matrix needed in ele
            # (remember it's a broadcast on element-basis)
            temp = np.empty((self.ndims, self.N, ele.neles))

            np.copyto(temp, self.eddies_loc[...,np.newaxis])
            ele.eddies_loc.set(temp)

            np.copyto(temp, self.eddies_strength[...,np.newaxis])
            ele.eddies_strength.set(temp)

            temp = np.empty((self.N, ele.neles))

            np.copyto(temp, self.eddies_time[...,np.newaxis])
            ele.eddies_time.set(temp)


    def __call__(self, intg):
        # Return if not active or no action is due
        if intg.nacptsteps % self.nsteps:
            return

        t = intg.tcurr

        # Check whether the eddies are outside of the box. If so generate new ones
        eddies_xl = self.eddies_loc[self.Ubulkdir] + (t - self.eddies_time)*self.Ubulk
        neweddies = eddies_xl > self.box_xmax

        if neweddies.any():
            self.create_eddies(t, neweddies)
