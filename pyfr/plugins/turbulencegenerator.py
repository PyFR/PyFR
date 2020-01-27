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

        dhat = d*self.Ubulk
        dhat /= c[...,np.newaxis]

        # source term and terms dependent on the location of solution/quadrature
        # points
        self.eles_turbsrc = []
        self.eles_dhatxhat = []
        self.eles_factor = []

        for etype, ele in elemap.items():
            # only for the momentum equations, so its dimensions are the same as self.ndims
            divfluxaa = 'div-flux' in ele.antialias
            pname = 'qpts' if divfluxaa else 'upts'

            npts = ele.nqpts if divfluxaa else ele.nupts

            # Allocate the memory to compute and update the source term.
            turbsrc = np.zeros((npts, self.ndims, ele.neles)).swapaxes(0, 1).reshape((self.ndims, -1))
            self.eles_turbsrc.append(turbsrc)

            # Initialize the actual source term to zero.
            # ele.turbsrc.set(self.eles_turbsrc[-1])
            ele.turbsrc.set(np.zeros((npts, self.ndims, ele.neles)))

            # Pre-compute some terms needed later.
            ploc = ele.ploc_at_np(pname).swapaxes(0, 1).reshape((self.ndims, -1))

            xhat = 2.*np.pi*ploc/self.lturb[0,0] #TODO

            #n = nvertices, d = ndims, N = nmodes
            dhatxhat = np.einsum('Nd,dn->nN', dhat, xhat)

            self.eles_dhatxhat.append(dhatxhat)

            # the scaling factor in front of the fluctuations
            factor = np.empty(turbsrc.shape)
            factor[...] = np.sqrt(2.0/self.N)

            #TODO multiply by aij properly
            factor *= np.sqrt(self.reystress[0,0])*(1.0 - np.abs(ploc[1]))

            # Correct it for the source term formulation by Schmidt. TODO
            factor *= np.exp(-np.pi*(ploc[self.Ubulkdir]
                                    - self.ctr[self.Ubulkdir])**2/2./self.lturb[0,0]**2)
            factor /= self.tturb[:, np.newaxis]

            self.eles_factor.append(factor)

    def compute_turbsrc(self, ele, tsrc, dhatxhat, factor, t):
        # keep in mind the original shape of ploc and tsrc: npts, ndims, neles
        that = 2.0*np.pi*t/self.tturb[self.Ubulkdir]

        arg = dhatxhat + self.ome*that #n = nvertices, d=ndims, N=nmodes
        modes = np.einsum('dN,nN->dnN', self.p, np.cos(arg)) \
              + np.einsum('dN,nN->dnN', self.q, np.sin(arg))

        tsrc = factor*np.sum(modes, axis=-1) #ndim x nvert

        # shape back and return
        return tsrc.reshape((ele.ndims, -1, ele.neles)).swapaxes(1, 0)

    def __call__(self, intg):
        # Return if not active or no action is due
        if not self._isactive or intg.nacptsteps % self.nsteps:
            return

        t = intg.tcurr

        for ele, turbsrc, dhatxhat, factor in zip(self.elemap.values(),
                                                  self.eles_turbsrc,
                                                  self.eles_dhatxhat,
                                                  self.eles_factor):
            ele.turbsrc.set(self.compute_turbsrc(ele, turbsrc, dhatxhat, factor, t))
