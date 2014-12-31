# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseFluidElements(object):
    _privarmap = {2: ['rho', 'u', 'v', 'p'],
                  3: ['rho', 'u', 'v', 'w', 'p']}

    _convarmap = {2: ['rho', 'rhou', 'rhov', 'E'],
                  3: ['rho', 'rhou', 'rhov', 'rhow', 'E']}

    def _process_ics(self, ics):
        rho, p = ics[0], ics[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in ics[1:-1]]

        # Compute the energy
        gamma = self.cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in ics[1:-1])

        return [rho] + rhovs + [E]


class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def set_backend(self, backend, nscalupts):
        super(EulerElements, self).set_backend(backend, nscalupts)

        # Register our flux kernel
        backend.pointwise.register('pyfr.solvers.euler.kernels.tflux')

        # Template parameters for the flux kernel
        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                       c=self.cfg.items_as('constants', float))

        if 'flux' in self.antialias:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self.smat_at('qpts'),
                f=self._vect_qpts
            )
        else:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self.smat_at('upts'),
                f=self._vect_upts
            )
