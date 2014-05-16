# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseElements


class BaseAdvectionElements(BaseElements):
    @property
    def _scratch_bufs(self):
        if 'flux' in self.antialias:
            return {'scal_fpts', 'scal_qpts', 'vect_qpts'}
        elif 'div-flux' in self.antialias:
            return {'scal_fpts', 'vect_upts', 'scal_qpts'}
        else:
            return {'scal_fpts', 'vect_upts'}

    def set_backend(self, backend, nscal_upts):
        super(BaseAdvectionElements, self).set_backend(backend, nscal_upts)

        # Register pointwise kernels with the backend
        backend.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.negdivconf'
        )

        # What anti-aliasing options we're running with
        fluxaa = 'flux' in self.antialias
        divfluxaa = 'div-flux' in self.antialias

        # Interpolation from elemental points
        if fluxaa:
            self.kernels['disu'] = lambda: backend.kernel(
                'mul', self.opmat('M8'), self.scal_upts_inb,
                out=self._scal_fqpts
            )
        else:
            self.kernels['disu'] = lambda: backend.kernel(
                'mul', self.opmat('M0'), self.scal_upts_inb,
                out=self._scal_fpts
            )

        # Interpolations and projections to/from quadrature points
        if divfluxaa:
            self.kernels['tdivf_qpts'] = lambda: backend.kernel(
                'mul', self.opmat('M7'), self.scal_upts_outb,
                out=self._scal_qpts
            )
            self.kernels['divf_upts'] = lambda: backend.kernel(
                'mul', self.opmat('M9'), self._scal_qpts,
                out=self.scal_upts_outb
            )

        # First flux correction kernel
        if fluxaa:
            self.kernels['tdivtpcorf'] = lambda: backend.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M10'), self._vect_qpts,
                out=self.scal_upts_outb
            )
        else:
            self.kernels['tdivtpcorf'] = lambda: backend.kernel(
                'mul', self.opmat('M1 - M3*M2'), self._vect_upts,
                out=self.scal_upts_outb
            )

        # Second flux correction kernel
        self.kernels['tdivtconf'] = lambda: backend.kernel(
            'mul', self.opmat('M3'), self._scal_fpts, out=self.scal_upts_outb,
            beta=1.0
        )

        # Transformed to physical divergence kernel
        if divfluxaa:
            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconf', tplargs=dict(nvars=self.nvars),
                dims=[self.nqpts, self.neles], tdivtconf=self._scal_qpts,
                rcpdjac=self.rcpdjac_at('qpts')
            )
        else:
            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconf', tplargs=dict(nvars=self.nvars),
                dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
                rcpdjac=self.rcpdjac_at('upts')
            )
