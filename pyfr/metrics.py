from functools import cached_property

import numpy as np

from pyfr.cache import memoize
from pyfr.shapes import interp_pts
from pyfr.util import ndrange


class MeshMetrics:
    def __init__(self, basis, spts, curved):
        self.basis = basis
        self.spts = spts

        self.nspts, self.neles, self.ndims = spts.shape
        self.nmpts = basis.nmpts

        # Treat all elements up to the last curved one as curved
        if basis.order >= 2:
            self.ncurved = int(np.max(*np.nonzero(curved), initial=-1)) + 1
        else:
            self.ncurved = self.neles

    def _pts(self, name):
        return getattr(self.basis, name) if isinstance(name, str) else name

    @memoize
    def _jacop_at(self, name):
        pt = self._pts(name)

        # Metric basis with grid point (q<=p) or pseudo grid points (q>p)
        mbasis = self.basis.mbasis

        # Jacobian operator at these points
        jacop = np.rollaxis(mbasis.jac_nodal_basis_at(pt), 2)
        return jacop.reshape(-1, self.nmpts)

    @memoize
    def jac_at(self, name):
        # Physical locations of the pseudo grid points
        x = self.ploc_at('mpts')

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
        op = self.basis.mbasis.nodal_basis_at(self._pts(name))

        # Apply and reshape
        smats = op @ smats_mpts
        return smats.reshape(self.ndims, -1, self.ndims, self.neles)

    @memoize
    def rcpdjac_at(self, name):
        _, djacs_mpts = self._smats_djacs_mpts

        # Interpolation matrix to pts
        m0 = self.basis.mbasis.nodal_basis_at(self._pts(name))

        # Interpolate the djacs
        djac = m0 @ djacs_mpts

        if np.any(djac < -1e-8*np.max(np.abs(djac), axis=0)):
            raise RuntimeError('Negative mesh Jacobians detected')

        return 1.0 / djac

    @memoize
    def ploc_at(self, name):
        op = self.basis.sbasis.nodal_basis_at(self._pts(name))

        return interp_pts(op, self.spts).swapaxes(1, 2)

    @memoize
    def pnorm_at(self, name, fidx):
        # Reference normal of the face
        norm = self.basis.faces[fidx][2]

        # We need to compute |J|*[(J^{-1})^{T}.N] where J is the
        # Jacobian and N is the normal for each point. Using
        # J^{-1} = S/|J| where S are the smats, we have S^{T}.N.
        pnorm = np.einsum('i,ijkl->jlk', norm, self.smat_at(name))

        # Compute the magnitudes of these flux point normals
        mag_pnorm = np.einsum('...i,...i', pnorm, pnorm)

        # Check that none of these magnitudes are zero
        if np.any(mag_pnorm < 1e-24*np.max(mag_pnorm, axis=0)):
            raise RuntimeError('Zero face normals detected')

        return pnorm

    @cached_property
    def _smats_djacs_mpts(self):
        # Dimensions, number of elements and number of mpts
        ndims, neles, nmpts = self.ndims, self.neles, self.nmpts

        # Physical locations of the pseudo grid points
        x = self.ploc_at('mpts')

        # Jacobian operator at these points
        jacop = self._jacop_at('mpts')

        # Transpose to get (ndims, ndims, nmpts, neles)
        jac = self.jac_at('mpts').transpose(2, 3, 0, 1)

        smats = np.empty((ndims, nmpts, ndims, neles))

        if ndims == 2:
            a, b, c, d = jac[0, 0], jac[1, 0], jac[0, 1], jac[1, 1]

            smats[0, :, 0], smats[0, :, 1] = d, -b
            smats[1, :, 0], smats[1, :, 1] = -c, a

            djacs = a*d - b*c
        else:
            # Handle the curved elements first using the curl form of smats
            nc = self.ncurved
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

            # Cross product form of smats for linear elements
            jl = jac[..., nc:]
            for i, c in ndrange(3, 3):
                j, k = (i + 1) % 3, (i + 2) % 3
                a, b = (c + 1) % 3, (c + 2) % 3
                smats[i, :, c, nc:] = jl[j, a]*jl[k, b] - jl[j, b]*jl[k, a]

            # We note that J = [x0; x1; x2]
            x0, x1, x2 = jac

            # Exploit the fact that det(J) = x0 · (x1 ^ x2)
            djacs = 0
            for i in range(3):
                j, k = (i + 1) % 3, (i + 2) % 3
                djacs += x0[i]*(x1[j]*x2[k] - x1[k]*x2[j])

        return smats.reshape(ndims, nmpts, -1), djacs
