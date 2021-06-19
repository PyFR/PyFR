# -*- coding: utf-8 -*-

from pyfr.solvers.aceuler.elements import BaseACFluidElements
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements


class ACNavierStokesElements(BaseACFluidElements,
                             BaseAdvectionDiffusionElements):
    @staticmethod
    def grad_con_to_pri(cons, grad_cons, cfg):
        return grad_cons

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Register our flux kernels
        kprefix = 'pyfr.solvers.acnavstokes.kernels'
        self._be.pointwise.register(f'{kprefix}.tflux')
        self._be.pointwise.register(f'{kprefix}.tfluxlin')

        # Template parameters for the flux kernels
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs
        }

        # Common arguments
        if 'flux' in self.antialias:
            u = lambda s: self._slice_mat(self._scal_qpts, s)
            f = lambda s: self._slice_mat(self._vect_qpts, s)
            pts, npts = 'qpts', self.nqpts
        else:
            u = lambda s: self._slice_mat(self.scal_upts_inb, s)
            f = lambda s: self._slice_mat(self._vect_upts, s)
            pts, npts = 'upts', self.nupts

        # Mesh regions
        regions = self._mesh_regions

        if 'curved' in regions:
            self.kernels['tdisf_curved'] = lambda: self._be.kernel(
                'tflux', tplargs=tplargs, dims=[npts, regions['curved']],
                u=u('curved'), f=f('curved'),
                smats=self.smat_at(pts, 'curved')
            )

        if 'linear' in regions:
            upts = getattr(self, pts)
            self.kernels['tdisf_linear'] = lambda: self._be.kernel(
                'tfluxlin', tplargs=tplargs, dims=[npts, regions['linear']],
                u=u('linear'), f=f('linear'),
                verts=self.ploc_at('linspts', 'linear'), upts=upts
            )
