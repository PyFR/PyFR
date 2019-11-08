# -*- coding: utf-8 -*-

from pyfr.backends.base.kernels import ComputeMetaKernel
from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    @property
    def _scratch_bufs(self):
        bufs = {'scal_fpts', 'vect_fpts', 'vect_upts'}

        if 'flux' in self.antialias:
            bufs |= {'scal_qpts', 'vect_qpts'}
        elif 'div-flux' in self.antialias:
            bufs |= {'scal_qpts'}

        if self._soln_in_src_exprs:
            if 'div-flux' in self.antialias:
                bufs |= {'scal_qpts_cpy'}
            else:
                bufs |= {'scal_upts_cpy'}

        return bufs

    def set_backend(self, backend, nscalupts, nonce, intoff):
        super().set_backend(backend, nscalupts, nonce, intoff)

        slicem = self._slice_mat
        kernel = self._be.kernel
        kernels = self.kernels

        # Register pointwise kernels
        self._be.pointwise.register(
            'pyfr.solvers.baseadvecdiff.kernels.gradcoru'
        )

        kernels['_copy_fpts'] = lambda: kernel(
            'copy', self._vect_fpts.slice(0, self.nfpts), self._scal_fpts
        )
        kernels['tgradpcoru_upts'] = lambda: kernel(
            'mul', self.opmat('M4 - M6*M0'), self.scal_upts_inb,
            out=self._vect_upts
        )

        for s, neles in self._ext_int_sides:
            kernels['tgradcoru_upts_' + s] = lambda s=s: kernel(
                'mul', self.opmat('M6'),
                slicem(self._vect_fpts, s, 0, self.nfpts),
                out=slicem(self._vect_upts, s), beta=1.0
            )
            kernels['gradcoru_upts_' + s] = lambda s=s, neles=neles: kernel(
                'gradcoru', tplargs=dict(ndims=self.ndims, nvars=self.nvars),
                 dims=[self.nupts, neles],
                 smats=self.smat_at('upts', s),
                 rcpdjac=self.rcpdjac_at('upts', s),
                 gradu=slicem(self._vect_upts, s)
            )

            def gradcoru_fpts(s=s):
                nupts, nfpts = self.nupts, self.nfpts
                vupts, vfpts = self._vect_upts, self._vect_fpts

                # Exploit the block-diagonal form of the operator
                muls = [kernel('mul', self.opmat('M0'),
                               slicem(vupts, s, i*nupts, (i + 1)*nupts),
                               slicem(vfpts, s, i*nfpts, (i + 1)*nfpts))
                        for i in range(self.ndims)]

                return ComputeMetaKernel(muls)

            kernels['gradcoru_fpts_' + s] = gradcoru_fpts

        if 'flux' in self.antialias:
            def gradcoru_qpts():
                nupts, nqpts = self.nupts, self.nqpts
                vupts, vqpts = self._vect_upts, self._vect_qpts

                # Exploit the block-diagonal form of the operator
                muls = [self._be.kernel('mul', self.opmat('M7'),
                                        vupts.slice(i*nupts, (i + 1)*nupts),
                                        vqpts.slice(i*nqpts, (i + 1)*nqpts))
                        for i in range(self.ndims)]

                return ComputeMetaKernel(muls)

            kernels['gradcoru_qpts'] = gradcoru_qpts

        # Shock capturing
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'artificial-viscosity':
            tags = {'align'}

            # Register the kernels
            self._be.pointwise.register(
                'pyfr.solvers.baseadvecdiff.kernels.shocksensor'
            )

            # Obtain the scalar variable to be used for shock sensing
            shockvar = self.convarmap[self.ndims].index(self.shockvar)

            # Obtain the degrees of the polynomial modes in the basis
            ubdegs = [sum(dd) for dd in self.basis.ubasis.degrees]

            # Template arguments
            tplargs = dict(
                nvars=self.nvars, nupts=self.nupts, svar=shockvar,
                c=self.cfg.items_as('solver-artificial-viscosity', float),
                order=self.basis.order, ubdegs=ubdegs,
                invvdm=self.basis.ubasis.invvdm.T
            )

            # Allocate space for the artificial viscosity vector
            self.artvisc = self._be.matrix((1, self.neles),
                                           extent=nonce + 'artvisc', tags=tags)

            # Apply the sensor to estimate the required artificial viscosity
            kernels['shocksensor'] = lambda: self._be.kernel(
                'shocksensor', tplargs=tplargs, dims=[self.neles],
                u=self.scal_upts_inb, artvisc=self.artvisc
            )
        elif shock_capturing == 'none':
            self.artvisc = None
        else:
            raise ValueError('Invalid shock capturing scheme')

    def get_artvisc_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]
        return (self.artvisc.mid,)*nfp, (0,)*nfp, (eidx,)*nfp
