# -*- coding: utf-8 -*-

from pyfr.backends.base import ComputeMetaKernel
from pyfr.solvers.base import BaseElements
from pyfr.plugins.turbulencegenerator import (eval_expr, get_lturbref,
                                              computeNeddies)

import numpy as np


class BaseAdvectionElements(BaseElements):
    @property
    def _scratch_bufs(self):
        if 'flux' in self.antialias:
            bufs = {'scal_fpts', 'scal_qpts', 'vect_qpts'}
        else:
            bufs = {'scal_fpts', 'vect_upts'}

        if self._soln_in_src_exprs:
            bufs |= {'scal_upts_cpy'}

        return bufs

    @staticmethod
    def arr_to_str(arr):
        string = np.array2string(arr, separator=',')
        return string.replace('[', '{').replace(']', '}').replace('\n', '')

    @staticmethod
    def G(csi, sigma, GC, p, clip, csimax):
        output = ((1./sigma/np.sqrt(2.0*np.pi*GC))
                  * np.exp(-0.5*((csi)/sigma)**2))**p
        if clip:
            if np.size(csimax) > 1:
                idx = np.abs(csi) > csimax[:, np.newaxis]
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
        # length scale and the reference (max) one (a reference for each
        # direction). The clipping value for the reference (max) lengths is 1.
        ndims, nvertices = ploc.shape
        GCs = np.zeros((ndims, ndims, nvertices))

        # Compute the constants (one constant for all vertices at the time).
        for i in range(ndims):  # x,y,z
            for j in range(ndims):  # U,V,W
                lturb = eval_expr(lturbexpr[i][j], constants, ploc)

                # Clipping value: smaller than reference length means smaller
                # No less than cmm for stability reasons, no more than 1 to
                # avoid inconsistencies.
                csimax = np.maximum(np.minimum(lturb/lturbref[i], 1.0), cmm)

                args = [sigma, 1.0, 2.0, True, csimax]
                def funct(csi): return G(csi, *args)
                GCs[i, j] = 0.5*fixed_quad(funct, -1, 1, n=50)[0]

        GCsInv = 1.0/np.sqrt(2.0*np.pi*GCs)
        # ndims, ndims, nvertices -> ndims, nvertices
        gauss_const = np.prod(GCsInv, axis=0)/(sigma**ndims)

        return gauss_const

    def prepare_turbsrc(self, ploc):
        cfgsect = 'soln-plugin-turbulencegenerator'

        # Constant variables
        constants = self.cfg.items_as('constants', float)

        npts, ndims, neles = ploc.shape

        # Ac or compressible?
        self.srctplargs['system'] = 'ac' if self.system.startswith(
            'ac') else 'compr'

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
        self.srctplargs['N'] = N = computeNeddies(inflow, Ubulkdir, lturbref)

        # expressions for the turbulent lengths, needed here for evaluation
        lturbexpr = [[self.cfg.getexpr(cfgsect, f'l{i}{j}')
                      for j in range(ndims)] for i in range(ndims)]

        # same as before but needed as template args to the kernel, for which
        # we need to substitute some variables and functions. They could be
        # computed here and used as arguments to the kernel but I want to save
        # some memory.
        subs = self.cfg.items('constants')
        subs.update(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs.update(abs='fabs', pi=str(np.pi))

        lturbexprkernel = [[self.cfg.getexpr(cfgsect, f'l{i}{j}', subs=subs)
                            for j in range(ndims)] for i in range(ndims)]

        self.srctplargs['lturbex'] = lturbexprkernel

        # Allocate the memory for the eddies location and strength.
        self.eddies_loc = self._be.matrix((ndims, N), tags={'align'})
        self._set_external('eddies_loc',
                           'in broadcast fpdtype_t[{}][{}]'.format(ndims, N),
                           value=self.eddies_loc)

        self.eddies_strength = self._be.matrix((ndims, N), tags={'align'})
        self._set_external('eddies_strength',
                           'in broadcast fpdtype_t[{}][{}]'.format(ndims, N),
                           value=self.eddies_strength)

        # Variables needed by the compressible solver
        if not self.system.startswith('ac'):
            # variables for the density fluctuations (Strong Reynolds Analogy):
            # mean density and mean Mach number
            rhomeanex = self.cfg.getexpr(cfgsect, 'rhomean', subs=subs)
            Mmeanex = self.cfg.getexpr(cfgsect, 'Mmean', subs=subs)
            self.srctplargs['rhomeanex'] = rhomeanex
            self.srctplargs['Mmeanex'] = Mmeanex

            self.srctplargs['rhofluctfactor'] = (
                constants['gamma'] - 1.0)/Ubulk

        # Center point
        ctr = np.array(self.cfg.getliteral(cfgsect, 'center'))

        # physical location of the solution points
        ploc = ploc.swapaxes(1, 0).reshape(ndims, -1)

        # Gaussian constants they depend on the box dimensions.
        sigma = self.cfg.getfloat(cfgsect, 'sigma', 1.0)
        self.srctplargs['arg_const'] = -0.5/(sigma**2)
        self.srctplargs['lturbref'] = self.arr_to_str(lturbref)

        gauss_const = self.determine_gaussian_constants(self.G, sigma,
                                                        lturbexpr, lturbref,
                                                        constants, ploc, cmm)

        # allocate the memory for the gauss constants
        gcmat = self._be.const_matrix(
            gauss_const.reshape(ndims, npts, neles).swapaxes(0, 1))
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
        broadcast_to_ploc = any([isinstance(lxk, np.ndarray)
                                 for lxk in lxklist])
        if broadcast_to_ploc:
            for idxl, lxk in enumerate(lxklist):
                if not isinstance(lxk, np.ndarray):
                    lxklist[idxl] = np.full((ploc.shape[-1],), lxk)
            lxk = np.array(lxklist)
        else:
            lxk = np.array(lxklist).reshape((ndims, 1))

        # clip lxk for stability
        lxk = np.maximum(lxk, cmm*lturbref[:, np.newaxis])

        # Scaling factor
        dist = (ploc[Ubulkdir] - ctr[Ubulkdir])/lxk

        factor = np.exp(-0.5*np.pi*np.power(dist, 2.)) * \
            Ubulk/lxk  # ndims, nvertices

        fmat = self._be.const_matrix(
            factor.reshape(ndims, npts, neles).swapaxes(0, 1))
        self._set_external('factor',
                           'in fpdtype_t[{0}]'.format(ndims),
                           value=fmat)

        # Compute and save the target Reynolds stress. Assume symmetry and
        # periodicity in the z direction.
        # TODO hardcoded for the moment.
        # #              [[R_00, R_01, R_02],
        # #               [R_10, R_11, R_12],
        # #               [R_20, R_21, R_22]]
        reystressexpr = [self.cfg.getexpr(
            cfgsect, f'r{i}{i}', '0') for i in range(ndims)]
        reystressexpr.append(self.cfg.getexpr(cfgsect, 'r01', '0'))
        # [r00, r11, r22, r01=r10]
        reystress = np.zeros((4, ploc.shape[-1]))
        for idx, expr in enumerate(reystressexpr):
            reystress[idx] = eval_expr(expr, constants, ploc)

        # the first three components should be positive
        reystress[0:3] = np.maximum(reystress[0:3], 0.0)

        # The aij matrix
        # TODO NOTE HARDCODING DIRECTION AND SIZE, and uniformity in z
        aij = np.empty((4, ploc.shape[-1]))
        aij[0] = np.sqrt(reystress[0])  # R00
        aij[1] = reystress[3]/np.maximum(aij[0], 1e-12)  # R01
        aij[2] = np.sqrt(np.maximum(reystress[1] - aij[1]**2, 0.0))  # R11
        aij[3] = np.sqrt(reystress[2])  # R22

        aijmat = self._be.const_matrix(
            aij.reshape(4, npts, neles).swapaxes(0, 1))
        self._set_external('aij',
                           'in fpdtype_t[{0}]'.format(4),
                           value=aijmat)

        # Array to determine whether or not an element is actually affected by
        # the box of eddies. Less than zero means not affected, i.e. outside of
        # the box of eddies.
        affected = -np.ones((1, neles))
        ploc = ploc.reshape((ndims, npts, neles))

        delta = np.zeros(ndims)
        delta[Ubulkdir] = lturbref[Ubulkdir]
        dirs = [i for i in range(ndims) if i != Ubulkdir]
        delta[dirs] = 0.5*inflow + lturbref[dirs]
        x0 = ctr - delta
        x1 = ctr + delta

        # Determine which points are inside the box
        inside = np.ones(ploc.shape[1:], dtype=np.bool)
        for l, p, u in zip(x0, ploc, x1):
            inside &= (l <= p) & (p <= u)

        if np.sum(inside):
            # indices of the elements that have at least one solution point
            # inside the box
            inside_eles = np.any(inside, axis=0).nonzero()[0]
            affected[0, inside_eles] = +1.0

        affectedmat = self._be.const_matrix(affected, tags={'align'})
        self._set_external('affected',
                           'in broadcast-col fpdtype_t',
                           value=affectedmat)

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        kernels = self.kernels

        # Register pointwise kernels with the backend
        self._be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.negdivconf'
        )

        # What anti-aliasing options we're running with
        fluxaa = 'flux' in self.antialias

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

        # Synthetic turbulence generation via source term.
        if self._turbsrc:
            self.system = self.cfg.get('solver', 'system')
            # Points locations are always needed while the conserved variables
            # are needed only by the compressible solver.
            plocsrc = True
            solnsrc = solnsrc if self.system.startswith('ac') else True

            # Compute/allocate the memory for the other needed variables.
            self.prepare_turbsrc(self.ploc_at_np('upts'))

        # Interpolation from elemental points
        kernels['disu'] = lambda: self._be.kernel(
            'mul', self.opmat('M0'), self.scal_upts_inb,
            out=self._scal_fpts
        )

        if fluxaa:
            kernels['qptsu'] = lambda: self._be.kernel(
                'mul', self.opmat('M7'), self.scal_upts_inb,
                out=self._scal_qpts
            )

        # First flux correction kernel
        if fluxaa:
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M9'), self._vect_qpts,
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
                    'mul', self.opmat('M10'), self.scal_upts_inb,
                    out=self._scal_upts_temp
                )
                copy = self._be.kernel(
                    'copy', self.scal_upts_inb, self._scal_upts_temp
                )

                return ComputeMetaKernel([mul, copy])

            kernels['filter_soln'] = filter_soln
