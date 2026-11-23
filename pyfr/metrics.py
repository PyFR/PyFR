from functools import cached_property

import numpy as np

from pyfr.bases import get_basis
from pyfr.cache import memoize
from pyfr.quadrules import get_quadrule
from pyfr.shapes import BaseShape, apply_op, l2_proj_op, proj_pts
from pyfr.util import ndrange, subclass_where


def cross_smats_djacs(jac):
    # We note that J = [x0; x1; x2]
    x0 = jac[0]

    if len(jac) == 2:
        (a, b), (c, d) = jac

        smats = np.array([[d, -c], [-b, a]])
    else:
        smats = np.empty_like(jac)

        for i, c in ndrange(3, 3):
            j, k = (i + 1) % 3, (i + 2) % 3
            a, b = (c + 1) % 3, (c + 2) % 3
            smats[i, c] = jac[j, a]*jac[k, b] - jac[j, b]*jac[k, a]

    # Exploit the fact that det(J) = x0 · (x1 ^ x2)
    return smats, sum(x0*smats[0])


class MeshMetrics:
    def __init__(self, shape, spts, curved):
        self.shape = shape
        self.spts = spts

        self.nspts, self.neles, self.ndims = spts.shape
        self.nmpts = shape.nmpts

        # Treat all elements up to the last curved one as curved
        if shape.order >= 2 and shape.jac_exprs:
            self.ncurved = int(np.max(*np.nonzero(curved), initial=-1)) + 1
        else:
            self.ncurved = self.neles

    def _pts(self, name):
        return getattr(self.shape, name) if isinstance(name, str) else name

    @property
    def centred_spts(self):
        return self.spts - self.spts.mean(axis=0)

    @memoize
    def _jacop_at(self, name):
        pt = self._pts(name)

        # Metric basis with grid point (q<=p) or pseudo grid points (q>p)
        mbasis = self.shape.mbasis

        # If the basis is polynomial obtain the Jacobian operator directly
        if mbasis.type == 'polynomial':
            jacop = mbasis.jac_nodal_basis_at(pt)
        # Otherwise, we need to project the derivative modes onto the basis
        else:
            qr = get_quadrule(self.shape.name, qdeg=2*mbasis.order)
            pjac = np.einsum('ij,klj->kli', l2_proj_op(qr, mbasis),
                             mbasis.jac_nodal_basis_at(qr.pts))

            # Evaluate the projected derivatives at these points
            jacop = np.einsum('ij,klj->kli', mbasis.nodal_basis_at(pt), pjac)

        return np.rollaxis(jacop, 2).reshape(-1, self.nmpts)

    @memoize
    def jac_at(self, name):
        # Physical locations of the pseudo grid points
        x = self.centred_ploc_at('mpts')

        # Cast as a matrix multiply and apply to eles
        jac = self._jacop_at(name) @ x.reshape(self.nmpts, -1)

        # Reshape (npts*ndims, neles*ndims) => (npts, ndims, ndims, neles)
        jac = jac.reshape(-1, self.ndims, self.ndims, self.neles)

        # Transpose to get (npts, neles, ndims, ndims) for batched linalg
        return jac.transpose(0, 3, 1, 2)

    @memoize
    def smat_at(self, name):
        smats_mpts, _ = self._smats_djacs_mpts

        # Interpolation operator from mpts to our pts
        op = self.shape.mbasis.nodal_basis_at(self._pts(name))

        # Apply and reshape
        smats = op @ smats_mpts
        return smats.reshape(self.ndims, -1, self.ndims, self.neles)

    @memoize
    def rcpdjac_at(self, name):
        _, djacs_mpts = self._smats_djacs_mpts

        # Interpolation matrix to pts
        m0 = self.shape.mbasis.nodal_basis_at(self._pts(name))

        # Interpolate the djacs
        djac = m0 @ djacs_mpts

        if np.any(djac < -1e-8*np.max(np.abs(djac), axis=0)):
            raise RuntimeError('Negative mesh Jacobians detected')

        return 1.0 / djac

    @memoize
    def ploc_at(self, name):
        op = self.shape.sbasis.nodal_basis_at(self._pts(name))

        return apply_op(op, self.spts).swapaxes(1, 2)

    @memoize
    def centred_ploc_at(self, name):
        op = self.shape.sbasis.nodal_basis_at(self._pts(name))

        return apply_op(op, self.centred_spts).swapaxes(1, 2)

    @memoize
    def pnorm_at(self, name, fidx):
        # Reference normal of the face
        norm = self.shape.faces[fidx][2]

        # We need to compute |J|*[(J^{-1})^{T}.N] where J is the
        # Jacobian and N is the normal for each point. When the metric
        # basis is polynomial we can use J^{-1} = S/|J|, where S are
        # the smats, to obtain S^{T}.N.
        if self.shape.mbasis.type == 'polynomial':
            pnorm = np.einsum('i,ijkl->jlk', norm, self.smat_at(name))
        # Otherwise the smats on a face depend on the interior of the
        # element, so reconstruct the polynomial mapping on the face
        # and form the normals from it
        else:
            pnorm = self._face_pnorm_at(name, fidx)

        # Compute the magnitudes of these flux point normals
        mag_pnorm = np.einsum('...i,...i', pnorm, pnorm)

        # Check that none of these magnitudes are zero
        if np.any(mag_pnorm < 1e-24*np.max(mag_pnorm, axis=0)):
            raise RuntimeError('Zero face normals detected')

        return pnorm

    @cached_property
    def pnorm_fpts(self):
        return np.vstack([self.pnorm_at(self.shape.fpts[fp], fidx)
                          for fidx, fp in enumerate(self.shape.facefpts)])

    def _face_pnorm_at(self, name, fidx):
        kind, proj, norm = self.shape.faces[fidx]
        order = self.shape.mbasis.order

        # Find the face nodes of the metric basis and their reference locations
        fnodes = subclass_where(BaseShape, name=kind).std_ele(order)
        rnodes = proj_pts(proj, fnodes)

        # Invert the affine map from the face to the reference element
        p0, tang = self.shape.facemaps[fidx]
        st = (self._pts(name) - p0) @ np.linalg.pinv(tang)

        # Compute the physical locations of the face nodes
        op = self.shape.sbasis.nodal_basis_at(rnodes)
        x = apply_op(op, self.centred_spts)

        # Form the derivative operators at the face nodes and requested points
        fbasis = get_basis(kind, order, fnodes)
        dfn = fbasis.jac_nodal_basis_at(fnodes)
        dop = fbasis.jac_nodal_basis_at(st)

        # Form the normal in invariant curl form; JSC 26(3), 301-327, Eq. (37)
        w = np.cross(x, [apply_op(d.T, x) for d in dfn])
        pnorm = 0.5*(apply_op(dop[0].T, w[1]) - apply_op(dop[1].T, w[0]))

        # Scale to the magnitude and orientation of the reference normal
        rt = np.cross(*tang)
        return pnorm*(rt @ norm / (rt @ rt))

    @cached_property
    def _smats_djacs_mpts(self):
        # Dimensions and number of mpts
        ndims, nmpts = self.ndims, self.nmpts

        # Physical locations of the pseudo grid points
        x = self.centred_ploc_at('mpts')

        # Jacobian operator at these points
        jacop = self._jacop_at('mpts')

        # Transpose to get (ndims, ndims, nmpts, neles)
        jac = self.jac_at('mpts').transpose(2, 3, 0, 1)

        # Cross product form of smats for linear elements
        smats, djacs = cross_smats_djacs(jac)
        smats = smats.swapaxes(1, 2)

        if ndims == 3 and (nc := self.ncurved):
            # Handle the curved elements using the curl form of smats
            tt = np.empty((nmpts, ndims, nc))
            dtt = []

            for dx in jac[..., :nc]:
                # Compute x cross x_(chi)
                for i in range(3):
                    j, k = (i + 1) % 3, (i + 2) % 3
                    tt[:, i] = x[:, j, :nc]*dx[k] - x[:, k, :nc]*dx[j]

                # Jacobian of x cross x_(chi) at the pseudo grid points
                dt = jacop @ tt.reshape(nmpts, -1)
                dt = dt.reshape(nmpts, ndims, ndims, -1).swapaxes(0, 1)

                dtt.append(dt)

            # Kopriva's invariant form of smats; JSC 26(3), 301-327, Eq. (37)
            smats[0, ..., :nc] = 0.5*(dtt[2][1] - dtt[1][2])
            smats[1, ..., :nc] = 0.5*(dtt[0][2] - dtt[2][0])
            smats[2, ..., :nc] = 0.5*(dtt[1][0] - dtt[0][1])

        return smats.reshape(ndims, nmpts, -1), djacs
