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

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Number of Fourier modes
        self.N = self.cfg.getint(cfgsect, 'N')
        self.factor = np.sqrt(2.0/self.N)

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
        self.tturb = self.lturb[self.Ubulkdir,:]/self.Ubulk

        # Whether or not this plug-in should be active at all
        self._isactive = self.cfg.getbool(cfgsect, 'active')

        # Center point and normal to (i.e. the intended direction of)
        # the generating plane.
        self.ctr = np.array(self.cfg.getliteral(cfgsect, 'center'))
        self.dir = np.array(self.cfg.getliteral(cfgsect, 'direction'))

        # update frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Underlying elements class
        #self.elementscls = intg.system.elementscls

        # get the elements
        self.elemap = elemap = intg.system.ele_map

        # # Check if the system is incompressible
        # self._ac = intg.system.name.startswith('ac')

        # # Get the type and shape of each element in the partition
        # etypes = intg.system.ele_types
        # # shapes = (nupts, nvars, neles)

        # # Solution matrices indexed by element type
        # self.solns = dict(zip(etypes, intg.soln)) #no actual need to know which element
        #                                           # or maybe yes

        # source term and location of solution/quadrature points
        self.eles_turbsrc = []
        self.eles_ploc = []

        for etype, ele in elemap.items():
            # only for the momentum equations, so its dimensions are the same as self.ndims
            divfluxaa = 'div-flux' in ele.antialias
            pname = 'qpts' if divfluxaa else 'upts'

            self.eles_ploc.append(ele.ploc_at_np(pname))

            npts = ele.nqpts if divfluxaa else ele.nupts

            # Allocate the memory to compute and update the source term.
            self.eles_turbsrc.append(np.zeros((npts, self.ndims, ele.neles)))

            # Initialize the actual source term to zero.
            ele.turbsrc.set(self.eles_turbsrc[-1])

        # Compute the random field variables we are going to use. Set the seed
        # to make sure the random field is consistent among the processes.
        seed = int(intg.tcurr) + 1
        np.random.seed(seed)
        eta = np.random.normal(0, 1,   (self.N, self.ndims))
        np.random.seed(seed*2)
        csi = np.random.normal(0, 1,   (self.N, self.ndims))
        np.random.seed(seed*3)
        self.ome = np.random.normal(1, 1,   (self.N))
        np.random.seed(seed*4)
        d   = np.random.normal(0, 0.5, (self.N, self.ndims))

        # Pre-compute some variables needed later.
        cnum = np.zeros((self.N))
        cnum = 3.*np.einsum('lm,Nl,Nm->N',self.reystress, d, d)
        cden = 2.*np.sum(d**2., axis=-1)

        c = np.sqrt(cnum/cden) #shape: N

        self.p = np.cross(eta, d).swapaxes(0,1) #shape: N, ndims -> ndims, N
        self.q = np.cross(csi, d).swapaxes(0,1)

        self.dhat = d*self.Ubulk
        self.dhat /= c[...,np.newaxis]

    def compute_turbsrc(self, tsrc, ploc, t):
        # keep in mind the shape: npts, ndims, neles
        if tsrc.shape != ploc.shape:
            print('ERROR! this should not happen')
        npts, ndims, neles = ploc.shape

        # reshape and swapaxes to simplify the operations
        tsrc = tsrc.swapaxes(0, 1).reshape((ndims, -1))
        ploc = ploc.swapaxes(0, 1).reshape((ndims, -1))

        twicepi = 2.0*np.pi

        #TODO add more dependencies if needed.

        # that = twicepi*t/self.tturb #tauhat k
        that = twicepi*t/self.tturb[0]

        xhat = twicepi*ploc/self.lturb[0,0]

        arg = np.einsum('Nd,dn->nN', self.dhat, xhat) + self.ome*that #n = nvertices, d=ndims, N=nmodes
        modes = np.einsum('dN,nN->dnN', self.p, np.cos(arg)) \
              + np.einsum('dN,nN->dnN', self.q, np.sin(arg))

        tsrc_common = self.factor*np.sum(modes, axis=-1) #ndim x nvert

        #TODO multipli by aij
        aij = self.reystress # TODO do the proper computation
        tsrc = 0.01*tsrc_common

        # divide by the characteristic time, and multiply everything by the exp
        # function as in Schmidt
        tsrc /= self.tturb[:, np.newaxis]
        tsrc *= np.exp(-np.pi*(ploc[0] - self.ctr[0])**2/2./self.lturb[0,0]**2)

        # shape back and return
        tsrc = tsrc.reshape((ndims, npts, neles)).swapaxes(1, 0)
        ploc = ploc.reshape((ndims, npts, neles)).swapaxes(1, 0)

        return tsrc

    def __call__(self, intg):
        # Return if not active or no action is due
        if not self._isactive or intg.nacptsteps % self.nsteps:
            return

        t = intg.tcurr

        for ele_turbsrc, ele, ele_ploc in zip(self.eles_turbsrc,
                                              self.elemap.values(),
                                              self.eles_ploc):
            ele.turbsrc.set(self.compute_turbsrc(ele_turbsrc, ele_ploc, t))
