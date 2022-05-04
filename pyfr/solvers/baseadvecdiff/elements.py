# -*- coding: utf-8 -*-

from pyfr.backends.base.kernels import MetaKernel
from pyfr.polys import get_polybasis
from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    @property
    def _scratch_bufs(self):
        bufs = {'scal_fpts', 'vect_fpts', 'vect_upts'}

        if 'flux' in self.antialias:
            bufs |= {'scal_qpts', 'vect_qpts'}

        if self._soln_in_src_exprs:
            bufs |= {'scal_upts_cpy'}

        return bufs

    def set_backend(self, backend, nscalupts, nonce, linoff):
        super().set_backend(backend, nscalupts, nonce, linoff)

        kernel = self._be.kernel
        kprefix = 'pyfr.solvers.baseadvecdiff.kernels'
        slicem = self._slice_mat

        # Register our pointwise kernels
        self._be.pointwise.register(f'{kprefix}.gradcoru')
        self._be.pointwise.register(f'{kprefix}.gradcorulin')

        # Mesh regions
        regions = self._mesh_regions

        self.kernels['_copy_fpts'] = lambda: kernel(
            'copy', self._vect_fpts.slice(0, self.nfpts), self._scal_fpts
        )
        if self.basis.order > 0:
            self.kernels['tgradpcoru_upts'] = lambda uin: kernel(
                'mul', self.opmat('M4 - M6*M0'), self.scal_upts[uin],
                out=self._vect_upts
            )
        self.kernels['tgradcoru_upts'] = lambda: kernel(
            'mul', self.opmat('M6'), self._vect_fpts.slice(0, self.nfpts),
            out=self._vect_upts, beta=float(self.basis.order > 0)
        )

        # Template arguments for the physical gradient kernel
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'jac_exprs': self.basis.jac_exprs
        }

        if 'curved' in regions:
            self.kernels['gradcoru_upts_curved'] = lambda: kernel(
                'gradcoru', tplargs=tplargs,
                dims=[self.nupts, regions['curved']],
                gradu=slicem(self._vect_upts, 'curved'),
                smats=self.curved_smat_at('upts'),
                rcpdjac=self.rcpdjac_at('upts', 'curved')
            )

        if 'linear' in regions:
            self.kernels['gradcoru_upts_linear'] = lambda: kernel(
                'gradcorulin', tplargs=tplargs,
                dims=[self.nupts, regions['linear']],
                gradu=slicem(self._vect_upts, 'linear'),
                upts=self.upts, verts=self.ploc_at('linspts', 'linear')
            )

        def gradcoru_fpts():
            nupts, nfpts = self.nupts, self.nfpts
            vupts, vfpts = self._vect_upts, self._vect_fpts

            # Exploit the block-diagonal form of the operator
            muls = [kernel('mul', self.opmat('M0'),
                           vupts.slice(i*nupts, (i + 1)*nupts),
                           vfpts.slice(i*nfpts, (i + 1)*nfpts))
                    for i in range(self.ndims)]

            return MetaKernel(muls)

        self.kernels['gradcoru_fpts'] = gradcoru_fpts

        if 'flux' in self.antialias and self.basis.order > 0:
            def gradcoru_qpts():
                nupts, nqpts = self.nupts, self.nqpts
                vupts, vqpts = self._vect_upts, self._vect_qpts

                # Exploit the block-diagonal form of the operator
                muls = [self._be.kernel('mul', self.opmat('M7'),
                                        vupts.slice(i*nupts, (i + 1)*nupts),
                                        vqpts.slice(i*nqpts, (i + 1)*nqpts))
                        for i in range(self.ndims)]

                return MetaKernel(muls)

            self.kernels['gradcoru_qpts'] = gradcoru_qpts

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

            # Obtain the name, degrees, and order of our solution basis
            ubname = self.basis.ubasis.name
            ubdegs = self.basis.ubasis.degrees
            uborder = self.basis.ubasis.order

            # Obtain the degrees of a basis whose order is one lower
            lubdegs = get_polybasis(ubname, max(0, uborder - 1)).degrees

            # Compute the intersection
            ind_modes = [d not in lubdegs for d in ubdegs]

            # Template arguments
            tplargs_artvisc = dict(
                nvars=self.nvars, nupts=self.nupts, svar=shockvar,
                c=self.cfg.items_as('solver-artificial-viscosity', float),
                order=self.basis.order, ind_modes=ind_modes,
                invvdm=self.basis.ubasis.invvdm.T
            )

            # Allocate space for the artificial viscosity vector
            self.artvisc = self._be.matrix((1, self.neles),
                                           extent=nonce + 'artvisc', tags=tags)

            # Apply the sensor to estimate the required artificial viscosity
            self.kernels['shocksensor'] = lambda uin: self._be.kernel(
                'shocksensor', tplargs=tplargs_artvisc, dims=[self.neles],
                u=self.scal_upts[uin], artvisc=self.artvisc
            )
        elif shock_capturing == 'none':
            self.artvisc = None
        else:
            raise ValueError('Invalid shock capturing scheme')

    def get_artvisc_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]
        return (self.artvisc.mid,)*nfp, (0,)*nfp, (eidx,)*nfp
