from functools import cached_property

import numpy as np

from pyfr.writers.vtk.base import interpolate_pts


class BasePostProcAdapter:
    def __init__(self, ctx, pris, grad_pris, ploc, shape=None, spts=None):
        self.pris = pris
        self.grad_pris = grad_pris
        self.ploc = ploc
        self.cfg = ctx.cfg
        self.soln = ctx.soln
        self.ndims = ploc.shape[0]

        self._elementscls = ctx.elementscls
        self._shape = shape
        self._spts = spts

    @property
    def shape(self):
        return self._shape

    @property
    def spts(self):
        return self._spts


class VolumePostProcAdapter(BasePostProcAdapter):
    pass


class STLPostProcAdapter(BasePostProcAdapter):
    pass


class BoundaryPostProcAdapter(BasePostProcAdapter):
    def __init__(self, ctx, pris, grad_pris, ploc, shape, spts,
                 fidx, svpts, face_norm):
        super().__init__(ctx, pris, grad_pris, ploc, shape, spts)

        self._fidx = fidx
        self._svpts = svpts
        self._face_norm = face_norm

    @cached_property
    def _eles(self):
        return self._elementscls(type(self._shape), self._spts, self.cfg)

    @cached_property
    def pnorm(self):
        norm_tiled = np.tile(self._face_norm, (len(self._svpts), 1))
        pn = self._eles.pnorm_at(self._svpts, norm_tiled)

        return pn.transpose(2, 0, 1)

    @cached_property
    def normals(self):
        mag = np.sqrt(np.einsum('ijk,ijk->jk', self.pnorm, self.pnorm))
        return self.pnorm / mag[None]

    @cached_property
    def tau_wall(self):
        mu = self.cfg.getfloat('constants', 'mu')
        normals = self.normals

        grad_vel = np.stack(self.grad_pris[1:self.ndims + 1])
        tau_n = mu * np.einsum('ijkl,jkl->ikl',
                               grad_vel + grad_vel.swapaxes(0, 1), normals)

        tau_nn = np.einsum('ikl,ikl->kl', tau_n, tau_n)
        tau_nn2 = np.einsum('ikl,ikl->kl', tau_n, normals)**2

        return np.sqrt(tau_nn - tau_nn2)

    @cached_property
    def wall_dist(self):
        itype, proj, norm = self._shape.faces[self._fidx]
        norm_arr = np.array(norm, dtype=float)
        upts = self._shape.upts

        face_pt = np.array(proj(*([0]*(self._shape.ndims - 1))))

        t = (face_pt - upts) @ norm_arr / np.dot(norm_arr, norm_arr)
        upts_on_face = upts + t[:, None] * norm_arr

        upt_op = self._shape.sbasis.nodal_basis_at(upts)
        face_op = self._shape.sbasis.nodal_basis_at(upts_on_face)

        x_upt = interpolate_pts(upt_op, self._spts)
        x_face = interpolate_pts(face_op, self._spts)

        diff = x_upt - x_face
        dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))

        dist = np.where(dist > 1e-10, dist, np.inf)
        wdist = dist.min(axis=0)

        nsvpts = self.ploc.shape[1]
        return np.broadcast_to(wdist, (nsvpts, len(wdist)))
