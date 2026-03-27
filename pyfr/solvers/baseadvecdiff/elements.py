from pyfr.solvers.base.elements import ExportableField, inters_map
from pyfr.solvers.baseadvec import BaseAdvectionElements
from pyfr.solvers.base.elements import ExportableField


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    @property
    def _scratch_bufs(self):
        bufs = {'scal_fpts', 'vect_fpts', 'vect_upts'}

        if 'flux' in self.antialias:
            bufs |= {'scal_qpts', 'vect_qpts'}
        elif self.grad_fusion:
            bufs |= {'grad_upts'}

        if self.basis.fpts_in_upts:
            bufs |= {'comm_fpts'}
            if self.grad_fusion:
                bufs -= {'vect_fpts'}

        return bufs

    def set_backend(self, backend, nonce, linoff):
        super().set_backend(backend, nonce, linoff)

        # Ensure we point to the correct gradient array
        if self.basis.fpts_in_upts:
            if self.grad_fusion:
                self.get_vect_fpts_for_inters = self._get_grad_upts_for_inters
            else:
                self.get_vect_fpts_for_inters = self._get_vect_fpts_for_inters

        kernel, kernels = self._be.kernel, self.kernels
        kprefix = 'pyfr.solvers.baseadvecdiff.kernels'
        slicem, slicedk = self._slice_mat, self._make_sliced_kernel

        # Register our pointwise kernels
        self._be.pointwise.register(f'{kprefix}.gradcoru')

        # Mesh regions
        regions = self.mesh_regions

        if abs(self.cfg.getfloat('solver-interfaces', 'ldg-beta')) == 0.5:
            kernels['copy_fpts'] = lambda: kernel(
                'copy', self._comm_fpts, self._scal_fpts
            )

        if self.basis.order > 0:
            kernels['tgradpcoru_upts'] = lambda uin: kernel(
                'mul', self.opmat('M4 - M6*M0'), self.scal_upts[uin],
                out=self._grad_upts
            )
        kernels['tgradcoru_upts'] = lambda: kernel(
            'mul', self.opmat('M6'), self._comm_fpts,
            out=self._grad_upts, beta=float(self.basis.order > 0)
        )

        # Template arguments for the physical gradient kernel
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'jac_exprs': self.basis.jac_exprs
        }

        gradcoru_u = []
        if 'curved' in regions:
            gradcoru_u.append(lambda: kernel(
                'gradcoru', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nupts, regions['curved']],
                gradu=slicem(self._grad_upts, 'curved'),
                smats=self.curved_smat_at('upts'),
                rcpdjac=self.rcpdjac_at('upts', 'curved')
            ))
        if 'linear' in regions:
            gradcoru_u.append(lambda: kernel(
                'gradcoru', tplargs=tplargs | {'ktype': 'linear'},
                dims=[self.nupts, regions['linear']],
                gradu=slicem(self._grad_upts, 'linear'),
                upts=self.upts, verts=self.ploc_at('linspts', 'linear')
            ))

        kernels['gradcoru_u'] = lambda: slicedk(k() for k in gradcoru_u)

        if not self.grad_fusion or self.basis.order == 0:
            kernels['gradcoru_upts'] = kernels['gradcoru_u']

        def gradcoru_fpts():
            nupts, nfpts = self.nupts, self.nfpts
            vupts, vfpts = self._grad_upts, self._vect_fpts

            # Exploit the block-diagonal form of the operator
            muls = [kernel('mul', self.opmat('M0'),
                           vupts.slice(i*nupts, (i + 1)*nupts),
                           vfpts.slice(i*nfpts, (i + 1)*nfpts))
                    for i in range(self.ndims)]

            return self._be.unordered_meta_kernel(muls)

        # Elide the interpolation if possible
        if not (self.basis.fpts_in_upts and self.grad_fusion):
            kernels['gradcoru_fpts'] = gradcoru_fpts

        if 'flux' in self.antialias and self.basis.order > 0:
            def gradcoru_qpts():
                nupts, nqpts = self.nupts, self.nqpts
                vupts, vqpts = self._vect_upts, self._vect_qpts

                # Exploit the block-diagonal form of the operator
                muls = [kernel('mul', self.opmat('M7'),
                               vupts.slice(i*nupts, (i + 1)*nupts),
                               vqpts.slice(i*nqpts, (i + 1)*nqpts))
                        for i in range(self.ndims)]

                return self._be.unordered_meta_kernel(muls)

            kernels['gradcoru_qpts'] = gradcoru_qpts

        # Artificial viscosity defaults; populated by ArtificialViscosity
        self.artvisc_fpts = None
        self.vtx_view = None
        self.vtx_views = {}

        # Register exportable AV field with lazy getter (must be here
        # so plugins can discover it before commit)
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'artificial-viscosity':
            nverts = len(self.basis.linspts)
            self.export_fields.append(ExportableField(
                name='artvisc', shape=(nverts,),
                getter=lambda: self.artvisc_vtx_fn()
            ))

    @inters_map
    def _get_grad_upts_for_inters(self, eidxs, fidx):
        rmap = self.srtd_face_fpts[fidx][eidxs]
        fmap = self.basis.fpts_map_upts[rmap]
        return self._grad_upts.mid, fmap, self.nupts

