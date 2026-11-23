from pyfr.solvers.base.elements import inters_map
from pyfr.solvers.baseadvec import BaseAdvectionElements
from pyfr.solvers.baseadvecdiff import operators


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    has_grad_soln = True
    operatorscls = operators.BaseAdvectionDiffusionOperators

    @property
    def _scratch_bufs(self):
        bufs = {'scal_fpts', 'vect_fpts', 'vect_upts', 'comm_fpts'}

        if self.fluxaa:
            bufs |= {'scal_qpts', 'vect_qpts'}
        elif self.grad_fusion:
            bufs |= {'grad_upts'}

        if self.shape.fpts_in_upts and self.grad_fusion:
            bufs -= {'vect_fpts'}

        return bufs

    def set_backend(self, backend, nonce):
        super().set_backend(backend, nonce)

        # Ensure we point to the correct gradient array
        if self.shape.fpts_in_upts:
            if self.grad_fusion:
                self.get_vect_fpts_for_inters = self._get_grad_upts_for_inters
            else:
                self.get_vect_fpts_for_inters = self._get_vect_fpts_for_inters

        kernel, kernels = self._be.kernel, self.kernels
        kprefix = 'pyfr.solvers.baseadvecdiff.kernels'

        # Register our pointwise kernels
        self._be.pointwise.register(f'{kprefix}.gradcoru_phys')

        # Uncorrected gradient at the solution points
        if self.shape.order > 0:
            kernels['gradpcoru_upts'] = lambda uin: self.gradpcoru(
                self.scal_upts[uin], self._grad_upts
            )

        # Correct using the common solution jumps at the flux points
        kernels['gradcoru_upts'] = lambda: self.gradcoru(
            self._comm_fpts, self._grad_upts, add=self.shape.order > 0
        )

        # Take the corrected gradients to physical space
        kernels['gradcoru_phys'] = lambda: self.gradcoru_phys(self._grad_upts)

        if not self.grad_fusion or self.shape.order == 0:
            kernels['gradcoru_phys_upts'] = kernels['gradcoru_phys']

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
        if not (self.shape.fpts_in_upts and self.grad_fusion):
            kernels['gradcoru_fpts'] = gradcoru_fpts

        if self.fluxaa and self.shape.order > 0:
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

    @inters_map
    def _get_grad_upts_for_inters(self, eidxs, fidx):
        rmap = self.srtd_face_fpts[fidx][eidxs]
        fmap = self.shape.fpts_map_upts[rmap]
        return self._grad_upts.mid, fmap, self.nupts

