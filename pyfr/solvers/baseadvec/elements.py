# -*- coding: utf-8 -*-

from pyfr.backends.base import ComputeMetaKernel
from pyfr.solvers.base import BaseElements


class BaseAdvectionElements(BaseElements):
    @property
    def _scratch_bufs(self):
        if 'flux' in self.antialias:
            bufs = {'scal_fpts', 'scal_qpts', 'vect_qpts'}
        else:
            bufs = {'scal_fpts', 'vect_upts'}

        if self._soln_in_src_exprs:
            bufs |= {'scal_upts_cpy'}

        return bufs

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        kernels = self.kernels

        # Register pointwise kernels with the backend
        self._be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.negdivconf'
        )

        # What anti-aliasing options we're running with
        fluxaa = 'flux' in self.antialias

        # What the source term expressions (if any) are a function of
        plocsrc = self._ploc_in_src_exprs
        solnsrc = self._soln_in_src_exprs

        # Source term kernel arguments
        srctplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'srcex': self._src_exprs
        }

        # Interpolation from elemental points
        if fluxaa:
            kernels['disu'] = lambda: self._be.kernel(
                'mul', self.opmat('M8'), self.scal_upts_inb,
                out=self._scal_fqpts
            )
        else:
            kernels['disu'] = lambda: self._be.kernel(
                'mul', self.opmat('M0'), self.scal_upts_inb,
                out=self._scal_fpts
            )

        # First flux correction kernel
        if fluxaa:
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M10'), self._vect_qpts,
                out=self.scal_upts_outb
            )
        else:
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('M1 - M3*M2'), self._vect_upts,
                out=self.scal_upts_outb
            )

        # Second flux correction kernel
        kernels['tdivtconf'] = lambda: self._be.kernel(
            'mul', self.opmat('M3'), self._scal_fpts, out=self.scal_upts_outb,
            beta=1.0
        )

        # Transformed to physical divergence kernel + source term
        plocupts = self.ploc_at('upts') if plocsrc else None
        solnupts = self._scal_upts_cpy if solnsrc else None

        if solnsrc:
            kernels['copy_soln'] = lambda: self._be.kernel(
                'copy', self._scal_upts_cpy, self.scal_upts_inb
            )

        kernels['negdivconf'] = lambda: self._be.kernel(
            'negdivconf', tplargs=srctplargs,
            dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
            rcpdjac=self.rcpdjac_at('upts'), ploc=plocupts, u=solnupts
        )

        # In-place solution filter
        if self.cfg.getint('soln-filter', 'nsteps', '0'):
            def filter_soln():
                mul = self._be.kernel(
                    'mul', self.opmat('M11'), self.scal_upts_inb,
                    out=self._scal_upts_temp
                )
                copy = self._be.kernel(
                    'copy', self.scal_upts_inb, self._scal_upts_temp
                )

                return ComputeMetaKernel([mul, copy])

            kernels['filter_soln'] = filter_soln
