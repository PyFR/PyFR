from collections import namedtuple
from functools import cached_property

import numpy as np

from pyfr.plugins.common import get_elementscls
from pyfr.readers.native import block_names
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where

FaceInfo = namedtuple('FaceInfo', 'etype fidx svpts norm')


class PostProcData:
    kind = 'volume'

    def __init__(self, cfg, pvars, ploc=None):
        self.cfg = cfg
        self.pvars = pvars
        self.ploc = ploc
        self.fields = {}

    @classmethod
    def from_soln(cls, soln, samples, ploc, *args):
        # Name the primitive variables and any gradients and residuals
        ndims, cfg = len(ploc), soln.config
        privars = get_elementscls(cfg).privars(ndims, cfg)
        names = privars + block_names(privars, ndims, soln.blocks)

        return cls(cfg, dict(zip(names, samples)), ploc, *args)

    @property
    def ndims(self):
        return len(self.ploc)

    @property
    def has_grads(self):
        return any(n.startswith('grad_') for n in self.pvars)

    def __getitem__(self, name):
        return self.pvars[name]


class BoundaryPostProcData(PostProcData):
    kind = 'boundary'

    def __init__(self, cfg, pvars, ploc, elementscls, spts, finfo):
        super().__init__(cfg, pvars, ploc)

        self._elementscls = elementscls
        self._spts = spts
        self._finfo = finfo

    @cached_property
    def _shape(self):
        cls = subclass_where(BaseShape, name=self._finfo.etype)
        return cls(len(self._spts), self.cfg)

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
        dx = x_upt[:, None] - np.moveaxis(self.ploc, 0, -1)[None]

        # Distance along the local surface normal at each surface point
        dist = np.abs(np.einsum('ipek,kpe->ipe', dx, self.normals))

        return dist.min(axis=0)
