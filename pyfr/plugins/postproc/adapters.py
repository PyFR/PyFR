from collections import namedtuple
from functools import cached_property

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


FaceInfo = namedtuple('FaceInfo', 'etype fidx svpts')


class PostProcData:
    def __init__(self, source, pvars, ploc):
        self.source = source
        self.cfg = source.cfg
        self.pvars = pvars
        self.ploc = ploc
        self.fields = {}

    @property
    def ndims(self):
        return self.source.ndims

    @property
    def has_grads(self):
        return any(n.startswith('grad_') for n in self.pvars)

    def __getitem__(self, name):
        return self.pvars[name]


class BoundaryPostProcData(PostProcData):
    def __init__(self, source, pvars, ploc, spts, finfo):
        super().__init__(source, pvars, ploc)

        self._spts = spts
        self._finfo = finfo

    @cached_property
    def _shape(self):
        cls = subclass_where(BaseShape, name=self._finfo.etype)
        return cls(len(self._spts), self.cfg)

    @cached_property
    def _eles(self):
        elementscls = self.source.elementscls
        return elementscls(type(self._shape), self._spts, self.cfg)

    @cached_property
    def pnorm(self):
        svpts = self._finfo.svpts
        norm = self._shape.faces[self._finfo.fidx][2]
        norm_tiled = np.tile(norm, (len(svpts), 1))
        pn = self._eles.pnorm_at(svpts, norm_tiled)

        return pn.transpose(2, 0, 1)

    @cached_property
    def normals(self):
        return self.pnorm / np.linalg.norm(self.pnorm, axis=0)

    @cached_property
    def boundary_dist(self):
        # Reference-space distance identifies interior solution points
        shape = self._shape
        _, proj, norm = shape.faces[self._finfo.fidx]

        norm = norm / np.linalg.norm(norm)
        face_pt = proj(*([0]*(shape.ndims - 1)))
        t = (face_pt - shape.upts) @ norm

        # Physical positions of the interior solution points
        op = shape.sbasis.nodal_basis_at(shape.upts[t != 0])
        r = op @ self._spts.reshape(op.shape[1], -1)
        x_upt = r.reshape(op.shape[0], *self._spts.shape[1:])

        # Offset of each interior point from each surface point
        dx = x_upt[:, None] - self.ploc[None]

        # Distance along the local surface normal at each surface point
        dist = np.abs(np.einsum('ipek,kpe->ipe', dx, self.normals))

        return dist.min(axis=0)
