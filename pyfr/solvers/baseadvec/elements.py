# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseElements


class BaseAdvectionElements(BaseElements):
    def set_backend(self, backend, nscal_upts):
        super(BaseAdvectionElements, self).set_backend(backend, nscal_upts)

        # Register pointwise kernels with the backend
        backend.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.negdivconf'
        )

        # Specify the kernels we provide
        self.kernels['disu_fpts'] = lambda: backend.kernel(
            'mul', self.opmat('M0'), self.scal_upts_inb, out=self._scal_fpts
        )
        self.kernels['tdivtpcorf_upts'] = lambda: backend.kernel(
            'mul', self.opmat('M132'), self._vect_upts,
            out=self.scal_upts_outb
        )
        self.kernels['tdivtconf_upts'] = lambda: backend.kernel(
            'mul', self.opmat('M3'), self._scal_fpts, out=self.scal_upts_outb,
            beta=1.0
        )
        self.kernels['negdivconf_upts'] = lambda: backend.kernel(
            'negdivconf', tplargs=dict(nvars=self.nvars),
            dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
            rcpdjac=self.rcpdjac_at('upts')
        )
