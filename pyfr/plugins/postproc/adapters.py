from functools import cached_property

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


class BasePostProcAdapter:
    def __init__(self, ctx, vsoln, vpts, etype, spts, has_grads=False):
        self.cfg = ctx.cfg
        self.elementscls = ctx.elementscls
        self.etype = etype
        self.spts = spts
        self.ploc = vpts.transpose(2, 0, 1)
        self.ndims = self.ploc.shape[0]
        self.dtype = ctx.dtype
        self.has_grads = has_grads
        self.fields = {}

        # Primitive variables and gradients (views into vsoln)
        nvars = len(self.elementscls.privars(self.ndims, self.cfg))
        self.pris = [vsoln[:, i, :] for i in range(nvars)]

        self.grad_pris = None
        if has_grads:
            nd = self.ndims
            self.grad_pris = [
                vsoln[:, j:j + nd, :].transpose(1, 0, 2)
                for j in range(nvars, nvars + nvars*nd, nd)
            ]

    @cached_property
    def shape(self):
        return subclass_where(BaseShape, name=self.etype)(
            len(self.spts), self.cfg
        )


class VolumePostProcAdapter(BasePostProcAdapter):
    pass


class STLPostProcAdapter(BasePostProcAdapter):
    pass


class BoundaryPostProcAdapter(BasePostProcAdapter):
    def __init__(self, ctx, vsoln, vpts, spts, finfo, has_grads=False):
        super().__init__(ctx, vsoln, vpts, finfo.etype, spts, has_grads)

        self._fidx = finfo.fidx
        self._svpts = finfo.svpts
        self._face_norm = finfo.norm

    @cached_property
    def _eles(self):
        return self.elementscls(type(self.shape), self.spts, self.cfg)

    @cached_property
    def pnorm(self):
        norm_tiled = np.tile(self._face_norm, (len(self._svpts), 1))
        pn = self._eles.pnorm_at(self._svpts, norm_tiled)

        return pn.transpose(2, 0, 1)

    @cached_property
    def normals(self):
        return self.pnorm / np.linalg.norm(self.pnorm, axis=0)

    @cached_property
    def _min_upt_wall_dist_approx(self):
        _, proj, norm = self.shape.faces[self._fidx]
        upts = self.shape.upts

        norm = norm / np.linalg.norm(norm)
        face_pt = proj(*([0]*(self.shape.ndims - 1)))
        t = (face_pt - upts) @ norm
        upts_on_face = upts + t[:, None] * norm

        sbasis = self.shape.sbasis

        def interp(op):
            r = op @ self.spts.reshape(op.shape[1], -1)
            return r.reshape(op.shape[0], *self.spts.shape[1:])

        x_upt = interp(sbasis.nodal_basis_at(upts))
        x_face = interp(sbasis.nodal_basis_at(upts_on_face))

        dist = np.linalg.norm(x_upt - x_face, axis=2)

        return dist[t != 0].min(axis=0)


class GradBoundaryPostProcAdapter(BoundaryPostProcAdapter):
    @cached_property
    def tau_wall(self):
        mu = self.cfg.getfloat('constants', 'mu')
        normals = self.normals

        grad_vel = np.stack(self.grad_pris[1:self.ndims + 1])
        sij = grad_vel + grad_vel.swapaxes(0, 1)
        tau_n = mu * np.einsum('ijkl,jkl->ikl', sij, normals)

        nn = (tau_n * normals).sum(axis=0)
        tau_tang = tau_n - nn * normals
        return np.linalg.norm(tau_tang, axis=0)
