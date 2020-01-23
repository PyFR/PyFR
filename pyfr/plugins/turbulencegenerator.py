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

        comm, rank, root = get_comm_rank_root()

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Number of Fourier modes
        self.N = self.cfg.getint(cfgsect, 'N')

        # Input Reynolds stress
        self.reystress = np.array(self.cfg.getliteral(cfgsect, 'ReynoldsStress'))
        # Reystress = np.array([[R_11, R_21, R_31],
        #               [R_21, R_22, R_32],
        #               [R_31, R_32, R_33]])

        # characteristic lengths,3x3 matrix (XYZ x UVW)
        self.lturb = np.array(self.cfg.getliteral(cfgsect, 'lturb'))

        # Bulk velocity
        self.Ubulk = self.cfg.getfloat(cfgsect, 'Ubulk')

        # bulk velocity direction, either 0,1, or 2 (i.e. x,y,z)
        self.Ubulkdir = self.cfg.getint(cfgsect, 'Ubulk-dir')

        # Frozen turbulence hypothesis to get characteristic times in each vel.
        # component.
        self.tturb = lturb[Ubulkdir,:]/Ubulk

        # Whether or not this plug-in should be active at all
        self._isactive = self.cfg.getint(cfgsect, 'active')

        # Center point and normal to (i.e. the intended direction of)
        # the generating plane.
        self.ctr = np.array(self.cfg.getliteral(cfgsect, 'center'))
        self.dir = np.array(self.cfg.getliteral(cfgsect, 'direction'))

        # update frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Whether divergence anti-alias in on or not.
        divfluxaa = 'div-flux' in self.elementscls.antialias

        # get the elements
        elemap = intg.system.ele_map

        # # Check if the system is incompressible
        # self._ac = intg.system.name.startswith('ac')

        # # Get the type and shape of each element in the partition
        # etypes = intg.system.ele_types
        # # shapes = (nupts, nvars, neles)

        # # Solution matrices indexed by element type
        # self.solns = dict(zip(etypes, intg.soln)) #no actual need to know which element
        #                                           # or maybe yes

        # source term and location of solution/quadrature points
        self.turbsrc = []
        self.turbsrc_new = []

        pname = 'qpts' if divfluxaa else 'upts'
        self.ele_ploc = [e.ploc_at_np(pname) for e in elemap.values()]

        for etype, ele in elemap.items():
            # Tell the backend about the source term we are adding, only for the
            # momentum equations, so its dimensions are the same as self.ndims
            npts = ele.nqpts if divfluxaa else ele.nupts

            self.turbsrc.append(ele._be.matrix((self.ndims,
                                                npts,
                                                ele.neles)))

            ele._set_external('turbsrc', 'in fpdtype_t[{}]'.format(self.ndims),
                                value=self.turbsrc[-1])

            # Allocate the memory to compute and update the source term.
            self.turbsrc_new.append(np.zeros((self.ndims, npts, ele.neles)))

        # Compute the random field variables we are going to use. Set the seed
        # to make sure the random field is consistent among the processes.
        seed = int(intg.tcurr) + 1
        np.random.seed(seed)
        eta = np.random.normal(0, 1,   (self.N, self.ndims))
        np.random.seed(seed*2)
        csi = np.random.normal(0, 1,   (self.N, self.ndims))
        np.random.seed(seed*3)
        ome = np.random.normal(1, 1,   (self.N))
        np.random.seed(seed*4)
        d   = np.random.normal(0, 0.5, (self.N, self.ndims))




    def __call__(self, intg):
        # Return if not active or no action is due
        if not self._isactive or intg.nacptsteps % self.nsteps:
            return

        # TEst
        for turbsrc, turbsrc_new in zip(self.turbsrc, self.turbsrc_new):
            turbsrc.set(turbsrc_new)
