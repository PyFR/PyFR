from functools import cached_property

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


class PostProcData:
    def __init__(self, cfg, soln, pris, ploc):
        self.cfg = cfg
        self.soln = soln
        self.nvars = len(soln.fields)
        self.ndims = ploc.shape[0]
        self._pris = pris
        self.ploc = ploc
        self.fields = {}

    @property
    def has_grads(self):
        return len(self._pris) > self.nvars

    @cached_property
    def pris(self):
        return list(self._pris[:self.nvars])

    @cached_property
    def grad_pris(self):
        return np.split(self._pris[self.nvars:], self.nvars)

    @cached_property
    def mu(self):
        cfg = self.cfg
        mu_ref = cfg.getfloat('constants', 'mu')

        if cfg.get('solver', 'viscosity-correction', 'none') == 'sutherland':
            gamma = cfg.getfloat('constants', 'gamma')
            cpTref = cfg.getfloat('constants', 'cpTref')
            cpTs = cfg.getfloat('constants', 'cpTs')

            rho, p = self.pris[0], self.pris[-1]
            cpT = gamma * p / ((gamma - 1) * rho)
            Trat = cpT / cpTref
            return (mu_ref * (cpTref + cpTs) * Trat * np.sqrt(Trat)
                    / (cpT + cpTs))
        else:
            return mu_ref


class BoundaryPostProcData(PostProcData):
    def __init__(self, cfg, soln, pris, ploc, elementscls, spts, finfo):
        super().__init__(cfg, soln, pris, ploc)

        self._elementscls = elementscls
        self._spts = spts
        self._finfo = finfo

    @cached_property
    def _shape(self):
        return subclass_where(BaseShape, name=self._finfo.etype)(
            len(self._spts), self.cfg
        )

    @cached_property
    def _eles(self):
        return self._elementscls(type(self._shape), self._spts, self.cfg)

    @cached_property
    def pnorm(self):
        svpts = self._finfo.svpts
        norm_tiled = np.tile(self._finfo.norm, (len(svpts), 1))
        pn = self._eles.pnorm_at(svpts, norm_tiled)

        return pn.transpose(2, 0, 1)

    @cached_property
    def normals(self):
        return self.pnorm / np.linalg.norm(self.pnorm, axis=0)

    @cached_property
    def min_upt_wall_dist_approx(self):
        shape = self._shape
        _, proj, norm = shape.faces[self._finfo.fidx]
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

    @cached_property
    def tau_wall(self):
        normals = self.normals

        grad_vel = np.stack(self.grad_pris[1:self.ndims + 1])
        sij = grad_vel + grad_vel.swapaxes(0, 1)
        tau_n = self.mu * np.einsum('ijkl,jkl->ikl', sij, normals)

        nn = (tau_n * normals).sum(axis=0)
        tau_tang = tau_n - nn * normals
        return np.linalg.norm(tau_tang, axis=0)
