# -*- coding: utf-8 -*-

from pyfr.backends.base import ComputeMetaKernel
from pyfr.solvers.base import BaseElements

import numpy as np

class BaseAdvectionElements(BaseElements):
    @property
    def _scratch_bufs(self):
        if 'flux' in self.antialias:
            bufs = {'scal_fpts', 'scal_qpts', 'vect_qpts'}
        elif 'div-flux' in self.antialias:
            bufs = {'scal_fpts', 'vect_upts', 'scal_qpts'}
        else:
            bufs = {'scal_fpts', 'vect_upts'}

        if self._soln_in_src_exprs:
            if 'div-flux' in self.antialias:
                bufs |= {'scal_qpts_cpy'}
            else:
                bufs |= {'scal_upts_cpy'}

        return bufs

    @staticmethod
    def arr_to_str(arr):
        string = np.array2string(arr, separator=',')
        return string.replace('[','{').replace(']','}').replace('\n','')

    @staticmethod
    def G(csi, sigma, GC, p, clip, csimax):
        output = ((1./sigma/np.sqrt(2.0*np.pi*GC))*np.exp(-0.5*((csi)/sigma)**2))**p
        if clip:
            output[np.abs(csi) > csimax] = 0.0
        return output

    @staticmethod
    def determine_gaussian_constants(G, sigma, ndims, lturb, Ubulkdir):
        from scipy.integrate import fixed_quad
        # Compute the constants GC such that 0.5 times the integral of the
        # guassian squared, between -1 and +1, is 1.0. The clipping value of
        # the independent variable depend on the ratio between the considered
        # length scale and the reference (max) one (a reference for each direction).
        # The clipping value for the reference (max) lengths is 1.
        GCs = np.zeros((ndims, ndims))

        # Reference lenghts, the maximum one per direction.
        lturbref = np.max(lturb, axis=1)

        # Clipping value: smaller than reference length means smaller clipping
        # value.
        csimax = lturb/lturbref[:,np.newaxis]

        # Compute the constants
        for i in range(ndims): #x,y,z
            for j in range(ndims): #U,V,W
                cm = csimax[i, j]
                args = [sigma, 1.0, 2.0, True, cm]
                funct = lambda csi: G(csi, *args)
                GCs[i,j] = 0.5*fixed_quad(funct, -1, 1, n=50)[0]

        return csimax, GCs, lturbref

    def prepare_turbsrc(self, ploc):
        cfgsect = 'soln-plugin-turbulencegenerator'

        # Constant variables
        constants = self.cfg.items_as('constants', float)

        npts, ndims, neles = ploc.shape

        # Ac or compressible?
        self.srctplargs['system'] = 'ac' if self.system.startswith('ac') else 'compr'

        # characteristic lengths,3x3 matrix (XYZ x UVW).
        lturb = np.array(self.cfg.getliteral(cfgsect, 'lturb'))

        # Bulk velocity magnitude
        Ubulk = self.cfg.getfloat(cfgsect, 'Ubulk')

        # bulk velocity direction, either 0,1, or 2 (i.e. x,y,z)
        Ubulkdir = self.cfg.getint(cfgsect, 'Ubulk-dir')

        self.srctplargs['Ubulk'] = Ubulk
        self.srctplargs['Ubulkdir'] = Ubulkdir

        # Number of eddies.
        inflow = np.array(self.cfg.getliteral(cfgsect, 'plane-dimensions'))
        dirs = [i for i in range(ndims) if i != Ubulkdir]

        inflowarea = np.prod(inflow)
        eddyarea = 4.0*np.prod(np.max(lturb[dirs], axis=1)) # 2 Ly x 2 Lz
        self.N = N = int(inflowarea/eddyarea) + 1
        print('n eddies = {}'.format(N))

        self.srctplargs['N'] = N

        # Gaussian constants they depend on the box dimensions.
        self.srctplargs['sigma'] = sigma = self.cfg.getfloat(cfgsect, 'sigma', 1.0)
        csimax, GCs, lturbref = self.determine_gaussian_constants(self.G, sigma, ndims, lturb, Ubulkdir)
        self.srctplargs['csimax'] = self.arr_to_str(csimax)
        self.srctplargs['GCs'] = self.arr_to_str(GCs)
        self.srctplargs['lturbref'] = self.arr_to_str(lturbref)

        # Allocate the memory for the eddies location and strength.
        self.eddies_loc = self._be.matrix((self.ndims, N))
        self._set_external('eddies_loc',
                           'in broadcast fpdtype_t[{}][{}]'.format(self.ndims, N),
                            value=self.eddies_loc)

        self.eddies_strength = self._be.matrix((self.ndims, N))
        self._set_external('eddies_strength',
                           'in broadcast fpdtype_t[{}][{}]'.format(self.ndims, N),
                            value=self.eddies_strength)

        #TODO compute the factor and aij mat in the plugin rather than here?

        # Frozen turbulence hypothesis to get characteristic times in each vel.
        # component. This is needed for the scaling factor
        tturb = lturb[Ubulkdir,:]/Ubulk

        # Center point
        ctr = np.array(self.cfg.getliteral(cfgsect, 'center'))

        # Scaling factor
        ploc = ploc.swapaxes(1, 0).reshape(ndims, -1)
        dist = (ploc[Ubulkdir] - ctr[Ubulkdir])/lturb[Ubulkdir,:,np.newaxis]

        factor = np.exp(-0.5*np.pi*np.power(dist, 2.))/tturb[:, np.newaxis] #ndims, nvertices

        # factor *= np.sqrt(1./N)

        fmat = self._be.const_matrix(factor.reshape(ndims, npts, neles).swapaxes(0, 1))
        self._set_external('factor',
                           'in fpdtype_t[{0}]'.format(ndims),
                           value=fmat)

        # Model the Reystress properly. TODO hard coded for the moment.
        # parametrized with respect to the non-dimensional distance from the wall
        utau2 = constants['tauw']/constants['rhom']
        ncoeffs = 11
        coeffs = np.empty((4,ncoeffs))
        #uRMS/utau -> R11
        coeffs[0,:] = [ 1.21715190e+04,-6.02428864e+04, 1.25234653e+05,-1.40761190e+05,
                        9.07187232e+04,-3.14035259e+04, 3.36630529e+03, 1.39816885e+03,
                       -5.47109999e+02, 6.61763704e+01, 0.0]
        #vRMS/utau -> R22
        coeffs[1,:] = [ 2.13721398e+03,-1.13401755e+04, 2.59234834e+04,-3.34149489e+04,
                        2.66892076e+04,-1.36591822e+04, 4.45867524e+03,-8.78495685e+02,
                        8.33507800e+01, 1.47856617e+00, 0.0]
        # wRMS/utau -> R33
        coeffs[2,:] = [-5.45793568e+03, 2.87861356e+04,-6.53703038e+04, 8.36764824e+04,
                       -6.64138439e+04, 3.39109628e+04,-1.12132925e+04, 2.37169831e+03,
                       -3.14605592e+02, 2.53135377e+01, 0.0]
        # minus avg(u' v')/utau**2 -> R12
        # note that this is fine for the lower part, should be negated for the upper part
        # as the vertical velocity, with respect to the distance from the wall, changes sign
        coeffs[3,:] = [ 7.46625281e+03,-3.96852213e+04, 9.07086776e+04,-1.16508846e+05,
                        9.21616093e+04,-4.62093468e+04, 1.45093738e+04,-2.68039137e+03,
                        2.40577760e+02,-2.67805765e+00, 0.0]

        # Special treatment for the mixed term
        reystress = np.zeros((4, ploc.shape[-1]))
        yc = ploc[1]/constants['delta']
        yw = 1.0 - np.abs(yc)
        for el in range(3):
            poly = np.poly1d(coeffs[el,:])
            reystress[el,:] = poly(yw)**2

        poly = np.poly1d(coeffs[3,:])
        reystress[3] = -np.poly1d(yw)  #MINUS as the coefficients are for -u'v'!
        # Correct the sign of the term for the upper part of the domain for
        # which v changes sign with respect to the closet wall (v is positive
        # when moving away from a wall).
        reystress[3][yc > 0.0] = -reystress[3][yc > 0.0]

        # Make them dimensional
        reystress *= utau2

        # #HARD CODE ARUP BOX
        # reystressmat = np.array(self.cfg.getliteral(cfgsect, 'ReynoldsStress'))
        # # Reystress = np.array([[R_11, R_21, R_31],
        # #               [R_21, R_22, R_32],
        # #               [R_31, R_32, R_33]])
        # reystress[0] = reystressmat[0,0]
        # reystress[1] = reystressmat[1,1]
        # reystress[2] = reystressmat[2,2]
        # reystress[3] = reystressmat[1,2]

        # The aij matrix
        #TODO NOTE HARDCODING DIRECTION AND SIZE, and uniformity in z
        aij = np.empty(reystress.shape)
        aij[0] = np.sqrt(reystress[0]) #R11
        aij[1] = reystress[3]/(aij[0] + 1e-10)   #R12
        aij[2] = np.sqrt(np.maximum(reystress[1] - aij[1]**2, 0.0)) #R22
        aij[3] = np.sqrt(reystress[2]) #R33

        aijmat = self._be.const_matrix(aij.reshape(4, npts, neles).swapaxes(0, 1))
        self._set_external('aij',
                           'in fpdtype_t[{0}]'.format(4),
                           value=aijmat)

        # Array to determine whether or not a sol point is actually affected by
        # the box of eddies. Zero means not affected, i.e. outside of the box
        # of eddies.
        affected = np.ones((npts, 1, neles)).reshape((-1))
        xmin = ctr[Ubulkdir] - np.max(lturb[Ubulkdir])
        xmax = ctr[Ubulkdir] + np.max(lturb[Ubulkdir])
        outside = np.logical_or(ploc[Ubulkdir] < xmin, ploc[Ubulkdir] > xmax)
        affected[outside] = -1.0
        affectedmat = self._be.const_matrix(affected.reshape((npts, 1, neles)))
        self._set_external('affected',
                           'in fpdtype_t[{0}]'.format(1),
                           value=affectedmat)


    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        slicem = self._slice_mat
        kernels = self.kernels

        # Register pointwise kernels with the backend
        self._be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.negdivconf'
        )

        # What anti-aliasing options we're running with
        fluxaa = 'flux' in self.antialias
        divfluxaa = 'div-flux' in self.antialias

        # What the source term expressions (if any) are a function of
        plocsrc = self._ploc_in_src_exprs
        solnsrc = self._soln_in_src_exprs

        # Source term kernel arguments
        self.srctplargs = srctplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'srcex': self._src_exprs
        }

        # External kernel arguments, if any.
        if self._turbsrc:
            self.system = self.cfg.get('solver', 'system')
            # Source term for turbulence generation.
            # We need the points locations and conserved variables for compr solver.
            plocsrc = True
            solnsrc = False if self.system.startswith('ac') else True

            # Compute/allocate the memory for the other needed variables.
            pname = 'qpts' if divfluxaa else 'upts'
            self.prepare_turbsrc(self.ploc_at_np(pname))

        # Interpolation from elemental points
        for s, neles in self._ext_int_sides:
            if fluxaa or (divfluxaa and solnsrc):
                kernels['disu_' + s] = lambda s=s: self._be.kernel(
                    'mul', self.opmat('M8'), slicem(self.scal_upts_inb, s),
                    out=slicem(self._scal_fqpts, s)
                )
            else:
                kernels['disu_' + s] = lambda s=s: self._be.kernel(
                    'mul', self.opmat('M0'), slicem(self.scal_upts_inb, s),
                    out=slicem(self._scal_fpts, s)
                )

        # Interpolations and projections to/from quadrature points
        if divfluxaa:
            kernels['tdivf_qpts'] = lambda: self._be.kernel(
                'mul', self.opmat('M7'), self.scal_upts_outb,
                out=self._scal_qpts
            )
            kernels['divf_upts'] = lambda: self._be.kernel(
                'mul', self.opmat('M9'), self._scal_qpts,
                out=self.scal_upts_outb
            )

        # First flux correction kernel
        if fluxaa:
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M10'), self._vect_qpts,
                out=self.scal_upts_outb
            )
        else:
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('M1 - M3*M2'), self._vect_upts,
                out=self.scal_upts_outb
            )

        # Second flux correction kernel
        kernels['tdivtconf'] = lambda: self._be.kernel(
            'mul', self.opmat('M3'), self._scal_fpts, out=self.scal_upts_outb,
            beta=1.0
        )

        # Transformed to physical divergence kernel + source term
        if divfluxaa:
            plocqpts = self.ploc_at('qpts') if plocsrc else None
            solnqpts = self._scal_qpts_cpy if solnsrc else None

            if solnsrc:
                kernels['copy_soln'] = lambda: self._be.kernel(
                    'copy', self._scal_qpts_cpy, self._scal_qpts
                )

            kernels['negdivconf'] = lambda: self._be.kernel(
                'negdivconf', tplargs=srctplargs,
                dims=[self.nqpts, self.neles], tdivtconf=self._scal_qpts,
                rcpdjac=self.rcpdjac_at('qpts'), ploc=plocqpts, u=solnqpts,
                extrns=self._external_args, **self._external_vals
            )
        else:
            plocupts = self.ploc_at('upts') if plocsrc else None
            solnupts = self._scal_upts_cpy if solnsrc else None

            if solnsrc:
                kernels['copy_soln'] = lambda: self._be.kernel(
                    'copy', self._scal_upts_cpy, self.scal_upts_inb
                )

            kernels['negdivconf'] = lambda: self._be.kernel(
                'negdivconf', tplargs=srctplargs,
                dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
                rcpdjac=self.rcpdjac_at('upts'), ploc=plocupts, u=solnupts,
                extrns=self._external_args, **self._external_vals
            )

        # In-place solution filter
        if self.cfg.getint('soln-filter', 'nsteps', '0'):
            def filter_soln():
                mul = self._be.kernel(
                    'mul', self.opmat('M11'), self.scal_upts_inb,
                    out=self._scal_upts_temp
                )
                copy = self._be.kernel(
                    'copy', self.scal_upts_inb, self._scal_upts_temp
                )

                return ComputeMetaKernel([mul, copy])

            kernels['filter_soln'] = filter_soln
