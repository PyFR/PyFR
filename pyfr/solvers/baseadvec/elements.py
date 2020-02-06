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

    def arr_to_str(self, arr):
        string = np.array2string(arr, separator=',')
        return string.replace('[','{').replace(']','}').replace('\n','')

    def prepare_turbsrc(self, ploc):
        cfgsect = 'solver-turbulencegenerator'

        # Constant variables
        constants = self.cfg.items_as('constants', float)

        npts, ndims, neles = ploc.shape

        # Update the template arguments with N, the number of Fourier modes
        self.srctplargs['N'] = N = self.cfg.getint('solver-turbulencegenerator',
                                              'N')

        #TODO need to tell the that it's matrix then set its value.
        # self.turbsrc = self._be.matrix((npts, self.ndims, self.neles))

        # Input Reynolds stress #TODO compute space dependent aij
        reystress = np.array(self.cfg.getliteral(cfgsect, 'ReynoldsStress'))
        # Reystress = np.array([[R_11, R_21, R_31],
        #               [R_21, R_22, R_32],
        #               [R_31, R_32, R_33]])

        # characteristic lengths,3x3 matrix (XYZ x UVW)
        lturb = np.array(self.cfg.getliteral(cfgsect, 'lturb'))

        self.srctplargs['ploc_scale'] = self.arr_to_str(np.pi*2.0/lturb)

        # Bulk velocity magnitude
        Ubulk = self.cfg.getfloat(cfgsect, 'Ubulk')

        # bulk velocity direction, either 0,1, or 2 (i.e. x,y,z)
        Ubulkdir = self.cfg.getint(cfgsect, 'Ubulk-dir')

        # Frozen turbulence hypothesis to get characteristic times in each vel.
        # component.
        tturb = lturb[Ubulkdir,:]/Ubulk

        self.srctplargs['t_scale'] = self.arr_to_str(np.pi*2/tturb)

        # Center point
        ctr = np.array(self.cfg.getliteral(cfgsect, 'center'))

        # random vars
        seed = 12346578
        np.random.seed(seed)
        eta = np.random.normal(0, 1,   (N, self.ndims))
        np.random.seed(seed*2)
        csi = np.random.normal(0, 1,   (N, self.ndims))
        np.random.seed(seed*3)
        ome = np.random.normal(1, 1,   (N))
        np.random.seed(seed*4)
        d   = np.random.normal(0, 0.5, (N, self.ndims))

        cnum = np.zeros((N))
        cnum = 3.*np.einsum('lm,Nl,Nm->N', reystress, d, d)
        cden = 2.*np.sum(d**2., axis=-1)

        c = np.sqrt(cnum/cden) #shape: N

        dhat = d*Ubulk
        dhat /= c[...,np.newaxis]

        p = np.cross(eta, d).swapaxes(0,1) #shape: N, ndims -> ndims, N
        q = np.cross(csi, d).swapaxes(0,1)

        # self._set_external('dhat',
        #                    'in broadcast fpdtype_t[{0}][{0}]'.format(ndims, N),
        #                    value=self._be.const_matrix(dhat))
        # self._set_external('p',
        #                    'in broadcast fpdtype_t[{0}][{0}]'.format(ndims, N),
        #                    value=self._be.const_matrix(p))
        # self._set_external('q',
        #                    'in broadcast fpdtype_t[{0}][{0}]'.format(ndims, N),
        #                    value=self._be.const_matrix(q))
        # self._set_external('ome',
        #                    'in broadcast fpdtype_t[{0}]'.format(N),
        #                    value=self._be.const_matrix(ome.reshape(1,N)))

        self.srctplargs['dhat'] = self.arr_to_str(dhat.swapaxes(0,1))
        self.srctplargs['p'] = self.arr_to_str(p)
        self.srctplargs['q'] = self.arr_to_str(q)
        self.srctplargs['ome'] = self.arr_to_str(ome)

        # Scaling factor
        ploc = ploc.swapaxes(1, 0).reshape(ndims, -1)
        dist = (ploc[Ubulkdir] - ctr[Ubulkdir])/lturb[Ubulkdir,:,np.newaxis]

        factor = np.exp(-0.5*np.pi*np.power(dist, 2.))/tturb[:, np.newaxis] #ndims, nvertices

        factor *= np.sqrt(2./N)

        # # TODO multiply by aij properly
        # factor *= np.sqrt(reystress[0,0])*(1.0 - np.abs(ploc[1]))

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
        if self.cfg.getint('solver-turbulencegenerator', 'N', 0):
            # Source term for turbulence generation.
            # We need the points locations, so modify plocsrc to make it
            # available in the kernel
            plocsrc = True
            solnsrc = True

            # Compute/Allocate the memory for the other needed variables.
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
