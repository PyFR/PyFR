from functools import cached_property, partial

from pyfr.backends.base import NullKernel
from pyfr.solvers.baseadvec.operators import (BaseAdvectionOperators,
                                              PhysicalAdvectionOperators,
                                              TransformedAdvectionOperators)


class BaseAdvectionDiffusionOperators(BaseAdvectionOperators):
    def gradpcoru(self, u, out):
        return self._be.kernel('mul', self.opmat('M4'), u, out=out)

    def gradcoru(self, ujump, out, add):
        return self._be.kernel(
            'mul', self.opmat('M6'), ujump, out=out, beta=float(add)
        )

    def grad_metrics(self, rgn):
        # Pair the kernel type of the gradients of a region with its metrics
        if rgn == 'curved':
            smats = self.curved_smat_at('upts')
            rcpdjac = self.rcpdjac_at('upts', 'curved')

            return 'curved', {'smats': smats, 'rcpdjac': rcpdjac}
        else:
            verts = self.centred_ploc_at('linspts', 'linear')

            return 'linear', {'verts': verts, 'upts': self.upts}

    @property
    def _gradcoru_phys_tplargs(self):
        # Template arguments for the physical gradient kernel
        return {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.shape.linspts),
            'jac_exprs': self.shape.jac_exprs
        }

    def gradcoru_phys(self, gradu):
        kernel, slicem = self._be.kernel, self._slice_mat
        tplargs, kerns = self._gradcoru_phys_tplargs, []

        for rgn, n in self.mesh_regions.items():
            ktype, kw = self.grad_metrics(rgn)

            kerns.append(kernel(
                'gradcoru_phys', tplargs=tplargs | {'ktype': ktype},
                dims=[self.nupts, n], gradu=slicem(gradu, rgn), **kw
            ))

        return self._make_sliced_kernel(kerns)


class TransformedAdvectionDiffusionOperators(BaseAdvectionDiffusionOperators,
                                             TransformedAdvectionOperators):
    pass


class PhysicalAdvectionDiffusionOperators(BaseAdvectionDiffusionOperators,
                                          PhysicalAdvectionOperators):
    physop_names = ('div', 'corr', 'grad')

    def set_backend(self, backend, nonce):
        super().set_backend(backend, nonce)

        # Register the kernel which corrects the gradients in physical space
        self._be.pointwise.register(
            'pyfr.solvers.baseadvecdiff.kernels.physgradapply'
        )

    def grad_metrics(self, rgn):
        # Leave the gradients of the curved elements in physical space
        if rgn == 'curved':
            return 'physical', {}
        else:
            return super().grad_metrics(rgn)

    @cached_property
    def _corr_pnorm(self):
        # Arrange the physical normals by direction for the correction kernel
        pn = self.metrics.pnorm_fpts[:, :self._nphyseles].transpose(2, 0, 1)

        return self._be.const_matrix(pn, tags={'align'})

    def _physgradapply(self, ujump, out, add):
        op = self._physops['corr']

        # Correct with the common solution jumps along each physical normal
        tplargs = {
            'ndims': self.ndims, 'nupts': self.nupts, 'ncol': op.ncol,
            'nvars': self.nvars, 'add': add
        }

        return self._be.kernel(
            'physgradapply', tplargs=tplargs, dims=[self._nphyseles],
            op=op, pnorm=self._corr_pnorm, ujump=ujump, out=out
        )

    def gradpcoru(self, u, out):
        pkern, lkern = partial(self._physapply, 'grad'), super().gradpcoru

        return self._make_region_kernel(pkern, lkern, u, out)

    def gradcoru(self, ujump, out, add):
        pkern, lkern = self._physgradapply, super().gradcoru

        return self._make_region_kernel(pkern, lkern, ujump, out, add=add)

    def gradcoru_phys(self, gradu):
        regions = self.mesh_regions

        # Skip the transformation if every gradient is formed in physical space
        if 'linear' not in regions:
            return NullKernel()
        # Transform as the reference operators do if no element is curved
        elif 'curved' not in regions:
            return super().gradcoru_phys(gradu)
        else:
            ktype, kw = self.grad_metrics('linear')
            tplargs = self._gradcoru_phys_tplargs | {'ktype': ktype}

            # Transform the gradients of the linear elements alone
            lkern = self._be.kernel(
                'gradcoru_phys', tplargs=tplargs,
                dims=[self.nupts, regions['linear']],
                gradu=self._slice_mat(gradu, 'linear'), **kw
            )

            return self._make_sliced_kernel([NullKernel(), lkern])
