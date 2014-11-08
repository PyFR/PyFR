# -*- coding: utf-8 -*-

import math
import re

from pyfr.backends.base.kernels import ComputeMetaKernel
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
            tplargs, plocqpts = self._process_src_terms('qpts')

            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconf', tplargs=tplargs, dims=[self.nqpts, self.neles],
                tdivtconf=self._scal_qpts, rcpdjac=self.rcpdjac_at('qpts'),
                ploc=plocqpts
            )
        else:
            tplargs, plocupts = self._process_src_terms('upts')

            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconf', tplargs=tplargs, dims=[self.nupts, self.neles],
                tdivtconf=self.scal_upts_outb,
                rcpdjac=self.rcpdjac_at('upts'), ploc=plocupts
            )

        # In-place solution filter
        if self.cfg.getint('soln-filter', 'freq', '0'):
            def filter_soln():
                mul = backend.kernel(
                    'mul', self.opmat('M11'), self.scal_upts_inb,
                    out=self._scal_upts_temp
                )
                copy = backend.kernel(
                    'copy', self.scal_upts_inb, self._scal_upts_temp
                )

                return ComputeMetaKernel([mul, copy])

            self.kernels['filter_soln'] = filter_soln

    def _process_src_terms(self, where):
        # Variable and function substitutions
        subs = self.cfg.items('constants')
        subs.update(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs.update(abs='fabs', pi=repr(math.pi))

        srcex = []
        for v in self._convarmap[self.ndims]:
            ex = self.cfg.get('solver-source-terms', v, '0')

            # Substitute variables
            ex = re.sub(r'\b({0})\b'.format('|'.join(subs)),
                        lambda m: subs[m.group(1)], ex)

            # Append
            srcex.append(ex)

        if any('ploc' in ex for ex in srcex):
            plocpts = self.ploc_at(where)
        else:
            plocpts = None

        return dict(ndims=self.ndims, nvars=self.nvars, srcex=srcex), plocpts
