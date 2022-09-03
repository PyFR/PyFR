# -*- coding: utf-8 -*-

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

    def set_backend(self, backend, nscalupts, nonce, linoff):
        super().set_backend(backend, nscalupts, nonce, linoff)

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
        kernels['disu'] = lambda uin: self._be.kernel(
            'mul', self.opmat('M0'), self.scal_upts[uin],
            out=self._scal_fpts
        )

        if fluxaa and self.basis.order > 0:
            kernels['qptsu'] = lambda uin: self._be.kernel(
                'mul', self.opmat('M7'), self.scal_upts[uin],
                out=self._scal_qpts
            )

        # First flux correction kernel
        if fluxaa and self.basis.order > 0:
            kernels['tdivtpcorf'] = lambda fout: self._be.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M9'), self._vect_qpts,
                out=self.scal_upts[fout]
            )
        elif self.basis.order > 0:
            kernels['tdivtpcorf'] = lambda fout: self._be.kernel(
                'mul', self.opmat('M1 - M3*M2'), self._vect_upts,
                out=self.scal_upts[fout]
            )

        # Second flux correction kernel
        kernels['tdivtconf'] = lambda fout: self._be.kernel(
            'mul', self.opmat('M3'), self._scal_fpts,
            out=self.scal_upts[fout], beta=float(self.basis.order > 0)
        )

        # Transformed to physical divergence kernel + source term
        plocupts = self.ploc_at('upts') if plocsrc else None
        solnupts = self._scal_upts_cpy if solnsrc else None

        if solnsrc:
            kernels['copy_soln'] = lambda uin: self._be.kernel(
                'copy', self._scal_upts_cpy, self.scal_upts[uin]
            )

        kernels['negdivconf'] = lambda fout: self._be.kernel(
            'negdivconf', tplargs=srctplargs,
            dims=[self.nupts, self.neles], tdivtconf=self.scal_upts[fout],
            rcpdjac=self.rcpdjac_at('upts'), ploc=plocupts, u=solnupts
        )

        # In-place solution filter
        if self.cfg.getint('soln-filter', 'nsteps', '0'):
            def filter_soln(uin):
                mul = self._be.kernel(
                    'mul', self.opmat('M10'), self.scal_upts[uin],
                    out=self._scal_upts_temp
                )
                copy = self._be.kernel(
                    'copy', self.scal_upts[uin], self._scal_upts_temp
                )

                return self._be.ordered_meta_kernel([mul, copy])

            kernels['filter_soln'] = filter_soln

        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'entropy-filter':
            tags = {'align'}

            # Allocate one minimum entropy value per interface
            self.nfaces = len(self.nfacefpts)
            ext = nonce + 'entmin_int'
            self.entmin_int = self._be.matrix((self.nfaces, self.neles),
                                               tags=tags, extent=ext)
   
            # Setup nodal/modal operator matrices
            self.vdm = self._be.const_matrix(self.basis.ubasis.vdm.T)
            self.invvdm = self._be.const_matrix(self.basis.ubasis.invvdm.T)
        else:
            self.entmin_int = None

    def get_entmin_int_fpts_for_inter(self, eidx, fidx):
        return (self.entmin_int.mid,), (fidx,), (eidx,)

    def get_entmin_bc_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]
        return (self.entmin_int.mid,)*nfp, (fidx,)*nfp, (eidx,)*nfp
