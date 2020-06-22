# -*- coding: utf-8 -*-

from pyfr.backends.base import ComputeMetaKernel
from pyfr.solvers.base import BaseElements
from pyfr.plugins.turbulencegenerator import eval_expr, get_lturbref

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
            if np.size(csimax) > 1:
                idx = np.abs(csi) > csimax[:,np.newaxis]
                output = np.copy(np.broadcast_to(output, idx.shape))
            else:
                idx = np.abs(csi) > csimax
            output[idx] = 0.0
        return output

    @staticmethod
    def determine_gaussian_constants(G, sigma, lturbexpr, lturbref, constants,
                                     ploc, cmm):
        from scipy.integrate import fixed_quad
        # Compute the constants GC such that 0.5 times the integral of the
        # guassian squared, between -1 and +1, is 1.0. The clipping value of
        # the independent variable depend on the ratio between the considered
        # length scale and the reference (max) one (a reference for each direction).
        # The clipping value for the reference (max) lengths is 1.
        ndims, nvertices = ploc.shape
        GCs = np.zeros((ndims, ndims, nvertices))

        # Compute the constants (one constant for all vertices at the time).
        for i in range(ndims): #x,y,z
            for j in range(ndims): #U,V,W
                lturb = eval_expr(lturbexpr[i][j], constants, ploc)

                # Clipping value: smaller than reference length means smaller
                # No less than cmm for stability reasons, no more than 1 to
                # avoid inconsistencies.
                csimax = np.maximum(np.minimum(lturb/lturbref[i], 1.0), cmm)

                args = [sigma, 1.0, 2.0, True, csimax]
                funct = lambda csi: G(csi, *args)
                GCs[i,j] = 0.5*fixed_quad(funct, -1, 1, n=50)[0]

        GCsInv = 1.0/np.sqrt(2.0*np.pi*GCs)
        # ndims, ndims, nvertices -> ndims, nvertices
        gauss_const = np.prod(GCsInv, axis=0)/(sigma**ndims)

        return gauss_const

    def prepare_turbsrc(self, ploc):
        import math
        cfgsect = 'soln-plugin-turbulencegenerator'

        # Constant variables
        constants = self.cfg.items_as('constants', float)

        npts, ndims, neles = ploc.shape

        # Ac or compressible?
        self.srctplargs['system'] = 'ac' if self.system.startswith('ac') else 'compr'

        # characteristic lengths,3x3 matrix (XYZ x UVW).
        # lturb = np.array(self.cfg.getliteral(cfgsect, 'lturb'))

        # minimum value of the clipping value csimax <= 1.0
        self.srctplargs['cmm'] = cmm = self.cfg.getfloat(cfgsect, 'csimax_min',
                                                         0.4)

        # reference turbulent lengths
        lturbref = get_lturbref(self.cfg, cfgsect, constants, ndims)

        # Bulk velocity
        Ubulk = self.cfg.getfloat(cfgsect, 'Ubulk')

        # bulk velocity direction, either 0,1, or 2 (i.e. x,y,z)
        Ubulkdir = self.cfg.getint(cfgsect, 'Ubulk-dir')

        self.srctplargs['Ubulkdir'] = Ubulkdir

        # Number of eddies.
        inflow = np.array(self.cfg.getliteral(cfgsect, 'plane-dimensions'))
        dirs = [i for i in range(ndims) if i != Ubulkdir]

        inflowarea = np.prod(inflow)
        eddyarea = 4.0*np.prod(lturbref[dirs]) # 2 Ly x 2 Lz
        self.N = N = int(inflowarea/eddyarea) + 1

        self.srctplargs['N'] = N

        # expressions for the turbulent lengths, needed here for evaluation
        lturbexpr = [[self.cfg.getexpr(cfgsect, f'l{i}{j}')
                      for j in range(ndims)] for i in range(ndims)]

        # same as before but needed as template args to the kernel, for which
        # we need to substitute some variables and functions. They could be
        # computed here and used as arguments to the kernel but I want to save
        # some memory.
        subs = self.cfg.items('constants')
        subs.update(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs.update(abs='fabs', pi=str(math.pi))

        lturbexprkernel = [[self.cfg.getexpr(cfgsect, f'l{i}{j}', subs=subs)
                            for j in range(ndims)] for i in range(ndims)]

        self.srctplargs['lturbex'] = lturbexprkernel

        # Allocate the memory for the eddies location and strength.
        self.eddies_loc = self._be.matrix((ndims, N))
        self._set_external('eddies_loc',
                           'in broadcast fpdtype_t[{}][{}]'.format(ndims, N),
                            value=self.eddies_loc)

        self.eddies_strength = self._be.matrix((ndims, N))
        self._set_external('eddies_strength',
                           'in broadcast fpdtype_t[{}][{}]'.format(ndims, N),
                            value=self.eddies_strength)

        #TODO compute the factor and aij mat in the plugin rather than here?

        # Frozen turbulence hypothesis to get characteristic times in each vel.
        # component. This is needed for the scaling factor
        # tturb = lturb[Ubulkdir,:]/Ubulk

        # Center point
        ctr = np.array(self.cfg.getliteral(cfgsect, 'center'))

        # physical location of the solution points
        ploc = ploc.swapaxes(1, 0).reshape(ndims, -1)

        # Gaussian constants they depend on the box dimensions.
        sigma = self.cfg.getfloat(cfgsect, 'sigma', 1.0)
        self.srctplargs['arg_const'] = -0.5/(sigma**2)
        self.srctplargs['lturbref'] = self.arr_to_str(lturbref)

        gauss_const = self.determine_gaussian_constants(self.G, sigma, lturbexpr,
                                                        lturbref, constants,
                                                        ploc, cmm)

        # allocate the memory for the gauss constants
        gcmat = self._be.const_matrix(gauss_const.reshape(ndims, npts, neles).swapaxes(0, 1))
        self._set_external('gauss_const',
                           'in fpdtype_t[{0}]'.format(ndims),
                           value=gcmat)

        # the actual values of the of the turbulent lengths. Each component
        # of the nested lists is a np.array of size nvertices=npts*neles
        # or a float if that component's lengths is constant
        lxklist = [eval_expr(lturbexpr[Ubulkdir][j], constants, ploc)
                   for j in range(ndims)]

        # if any is a numpy array then all components need to be too.
        # otherwise they are all constant and lxk just needs to be reshaped.
        broadcast_to_ploc = any([isinstance(lxk, np.ndarray) for lxk in lxklist])
        if broadcast_to_ploc:
            for idxl,lxk in enumerate(lxklist):
                if not isinstance(lxk, np.ndarray):
                    lxklist[idxl] = np.full((ploc.shape[-1],), lxk)
            lxk = np.array(lxklist)
        else:
            lxk = np.array(lxklist).reshape((ndims, 1))
        # print('ploc shape = {}'.format(ploc.shape) + 'lxk shape = {}'.format(lxk.shape))

        #clip lxk for stability
        lxk = np.maximum(lxk, cmm*lturbref[:, np.newaxis])

        # Scaling factor
        dist = (ploc[Ubulkdir] - ctr[Ubulkdir])/lxk
        # print('ploc[Ubulkdir] shape = {}'.format(ploc[Ubulkdir].shape))
        # print('dist shape = {}'.format(dist.shape))

        factor = np.exp(-0.5*np.pi*np.power(dist, 2.))*Ubulk/lxk #ndims, nvertices
        # print('factor shape = {}'.format(factor.shape))

        # factor *= np.sqrt(1./N)

        fmat = self._be.const_matrix(factor.reshape(ndims, npts, neles).swapaxes(0, 1))
        self._set_external('factor',
                           'in fpdtype_t[{0}]'.format(ndims),
                           value=fmat)

        # Model the Reystress properly. TODO hard coded for the moment.
        # parametrized with respect to the non-dimensional distance from the wall y/delta
        utau2 = constants['utau']**2 #constants['tauw']/constants['rhom']
        ncoeffs = 15
        coeffs = np.empty((4,ncoeffs))
        #uRMS/utau -> R11
        coeffs[0,:] = [ 2.19486972e+06,-1.58437492e+07, 5.12883686e+07,-9.82802707e+07,
                        1.23921817e+08,-1.08083933e+08, 6.67008924e+07,-2.92587482e+07,
                        9.01707856e+06,-1.88954813e+06, 2.50183446e+05,-1.70298457e+04,
                        1.26582088e+00, 6.87492352e+01,-1.48027907e-02]
        #vRMS/utau -> R22
        coeffs[1,:] = [ 1.15374190e+05,-8.35012705e+05, 2.71867287e+06,-5.26252656e+06,
                        6.74359493e+06,-6.02882218e+06, 3.86113761e+06,-1.79096772e+06,
                        6.01467524e+05,-1.44772848e+05, 2.44335085e+04,-2.74969383e+03,
                        1.70501429e+02, 1.19079597e+00,-2.40344727e-03]
        # wRMS/utau -> R33
        coeffs[2,:] = [-4.67724204e+05, 3.35837137e+06,-1.08218347e+07, 2.06721268e+07,
                       -2.60506922e+07, 2.28090039e+07,-1.42375116e+07, 6.40050289e+06,
                       -2.06998191e+06, 4.76610822e+05,-7.67311913e+04, 8.46495414e+03,
                       -6.38190408e+02, 3.37641881e+01, 3.62081709e-03]
        # avg(u' v')/utau**2 -> R12
        # note that this is fine for the lower part, should be negated for the upper part
        # as the vertical velocity, with respect to the distance from the wall, changes sign
        coeffs[3,:] = [ 1.43887186e+05,-9.24421506e+05, 2.55832472e+06,-3.91222486e+06,
                        3.42321944e+06,-1.36252894e+06,-4.23777124e+05, 9.08513536e+05,
                       -5.83322471e+05, 2.15251323e+05,-4.91871036e+04, 6.72652324e+03,
                       -4.66587078e+02, 5.87033236e+00,-1.23273856e-02]

        # Special treatment for the mixed term
        reystress = np.zeros((4, ploc.shape[-1]))
        yc = ploc[1]/constants['delta']
        yw = 1.0 - np.abs(yc)
        for el in range(4):
            poly = np.poly1d(coeffs[el,:])
            reystress[el,:] = poly(yw) if el==3 else poly(yw)**2

        # Correct the sign of the term for the upper part of the domain for
        # which v changes sign with respect to the closet wall (v is positive
        # when moving away from a wall).
        reystress[3][yc > 0.0] = -reystress[3][yc > 0.0]

        # Make them dimensional
        reystress *= utau2

        # the first three components should be positive
        reystress[0:3] = np.maximum(reystress[0:3], 0.0)

        # #HARD CODE ARUP BOX
        # reystressmat = np.array(self.cfg.getliteral(cfgsect, 'ReynoldsStress'))
        # # Reystress = np.array([[R_11, R_21, R_31],
        # #               [R_21, R_22, R_32],
        # #               [R_31, R_32, R_33]])
        # reystress[0] = reystressmat[0,0]
        # reystress[1] = reystressmat[1,1]
        # reystress[2] = reystressmat[2,2]
        # reystress[3] = reystressmat[0,1]

        # The aij matrix
        #TODO NOTE HARDCODING DIRECTION AND SIZE, and uniformity in z
        aij = np.empty((4, ploc.shape[-1]))
        aij[0] = np.sqrt(reystress[0]) #R11
        aij[1] = reystress[3]/np.maximum(aij[0], 1e-12)   #R12
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
        xmin = ctr[Ubulkdir] - lturbref[Ubulkdir]
        xmax = ctr[Ubulkdir] + lturbref[Ubulkdir]
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
            'srcex': self._src_exprs,
            'trsrc': self._turbsrc
        }

        # External kernel arguments for turbulence generation, if any.
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
