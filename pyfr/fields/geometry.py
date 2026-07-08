from functools import cached_property

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


class BoundaryGeometry:
    def __init__(self, cfg, spts, etype, fidx, svpts):
        self.cfg = cfg
        self._spts = spts
        self._etype = etype
        self._fidx = fidx
        self._svpts = svpts

    @cached_property
    def _shape(self):
        cls = subclass_where(BaseShape, name=self._etype)
        return cls(len(self._spts), self.cfg)

    @cached_property
    def _eles(self):
        from pyfr.solvers.base import BaseSystem

        sname = self.cfg.get('solver', 'system')
        ecls = subclass_where(BaseSystem, name=sname).elementscls

        return ecls(type(self._shape), self._spts, self.cfg)

    @cached_property
    def normals(self):
        _, _, norm = self._shape.faces[self._fidx]
        norm_tiled = np.tile(norm, (len(self._svpts), 1))
        pn = self._eles.pnorm_at(self._svpts, norm_tiled)
        pn = pn.transpose(2, 0, 1)

        return pn / np.linalg.norm(pn, axis=0)

    @cached_property
    def min_upt_wall_dist_approx(self):
        shape = self._shape
        _, proj, norm = shape.faces[self._fidx]
        upts = shape.upts

        norm /= np.linalg.norm(norm)
        face_pt = proj(*([0]*(shape.ndims - 1)))
        t = (face_pt - upts) @ norm
        upts_on_face = upts + t[:, None] * norm

        sbasis = shape.sbasis

        def interp(op):
            r = op @ self._spts.reshape(op.shape[1], -1)
            return r.reshape(op.shape[0], *self._spts.shape[1:])

        x_upt = interp(sbasis.nodal_basis_at(upts))
        x_face = interp(sbasis.nodal_basis_at(upts_on_face))

        dist = np.linalg.norm(x_upt - x_face, axis=2)

        return dist[t != 0].min(axis=0)

    def symbols(self, ndims):
        n = self.normals
        geom = {f'n_{x}': n[i] for i, x in enumerate('xyz'[:ndims])}
        geom['wall_dist'] = self.min_upt_wall_dist_approx

        return geom
