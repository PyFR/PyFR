# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    def set_backend(self, backend, nscalupts):
        super(NavierStokesElements, self).set_backend(backend, nscalupts)
        backend.pointwise.register('pyfr.solvers.navstokes.kernels.tflux')

        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                       c=self._cfg.items_as('constants', float))

        self.kernels['tdisf_upts'] = lambda: backend.kernel(
            'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
             u=self.scal_upts_inb, smats=self._smat_upts, f=self._vect_upts[0]
        )
