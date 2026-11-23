from functools import cached_property, partial
import sys

import numpy as np

from pyfr.cache import memoize
from pyfr.metrics import cross_smats_djacs
from pyfr.nputil import zip_chunks
from pyfr.quadrules import get_quadrule


class BaseAdvectionOperators:
    operators_name = None

    # Types of basis the operators can act on
    basis_types = ()

    def flux_metrics(self, rgn, pts):
        # Pair the kernel type of the fluxes of a region with its metric terms
        if rgn == 'curved':
            return 'curved', {'smats': self.curved_smat_at(pts)}
        else:
            verts = self.centred_ploc_at('linspts', 'linear')

            # Reference point coordinates for the linear smats
            return 'linear', {'verts': verts, 'upts': getattr(self, pts)}

    def divpcorf(self, f, out):
        expr = '(M1 - M3*M2)*M9' if self.fluxaa else 'M1 - M3*M2'

        return self._be.kernel('mul', self.opmat(expr), f, out=out)

    def divconf(self, fn, out, add):
        return self._be.kernel(
            'mul', self.opmat('M3'), fn, out=out, beta=float(add)
        )

    def div_metrics(self):
        # Scale the divergence by the reciprocal Jacobian determinants
        return 'transformed', {'rcpdjac': self.rcpdjac_at('upts')}


class TransformedAdvectionOperators(BaseAdvectionOperators):
    operators_name = 'transformed'
    basis_types = ('polynomial',)


class PhysicalAssembly:
    # Number of elements whose operators are assembled together
    chunksz = 1024

    def __init__(self, shape, metrics, nphyseles):
        self.shape, self.metrics = shape, metrics
        self.nphyseles, self.nupts = nphyseles, shape.nupts

        # Use a quadrature rule which is exact for the order-q geometry
        qdeg = 2*shape.order + shape.ndims*shape.nsptsord - 1
        qr = get_quadrule(shape.name, qdeg=qdeg)
        self._qpts, self._qwts = qr.pts, qr.wts

    @cached_property
    def _nodal_qpts(self):
        # Evaluate the nodal basis at the quadrature points
        return self.shape.ubasis.nodal_basis_at(self._qpts)

    @cached_property
    def _jac_nodal_qpts(self):
        # Evaluate the reference gradient of the basis at the quadrature points
        return self.shape.ubasis.jac_nodal_basis_at(self._qpts)

    @cached_property
    def _fmass(self):
        invvdm = self.shape.ubasis.invvdm

        # Form the face mass matrix from the correction functions
        return invvdm @ invvdm.T @ self.shape.m3

    @cached_property
    def _ref_laplacian_pinv(self):
        dL = self._jac_nodal_qpts

        # Form the weak Laplacian of the reference element
        lap = np.einsum('i,jki,jli->kl', self._qwts, dL, dL, optimize=True)

        return np.linalg.pinv(lap, rcond=1e-10, hermitian=True)

    @cached_property
    def _shape_jacop(self):
        # Evaluate the gradient of the shape basis at the quadrature points
        return self.shape.sbasis.jac_nodal_basis_at(self._qpts)

    def _smats_djacs_qpts(self, spts):
        spts = spts - spts.mean(axis=0)

        # Find the mapping Jacobian at the quadrature points of each element
        jac = np.einsum('ijk,jlm->imkl', self._shape_jacop, spts)

        # Form the transformation |J| J^{-T} of the reference gradients
        smat, djac = cross_smats_djacs(jac)

        if np.any(djac <= 0):
            raise RuntimeError('Negative mesh Jacobians detected')

        return smat, djac

    def _mass(self, djac):
        L = self._nodal_qpts

        return np.einsum('ij,ik,il->jkl', self._qwts[:, None]*djac, L, L,
                         optimize=('greedy', sys.maxsize))

    def _stiffness(self, smat):
        L, dL = self._nodal_qpts, self._jac_nodal_qpts

        # Form the stiffness matrices S_kij = (l_i, dl_j/dx_k) of each element
        return np.einsum('i,ij,kli,knim->mnjl', self._qwts, L, dL, smat,
                         optimize=('greedy', sys.maxsize))

    def _correct_stiffness(self, S, bnd):
        L, dL = self._nodal_qpts, self._jac_nodal_qpts

        # Compute the defect in the discrete divergence theorem
        d = bnd - S.sum(axis=2)

        # Correct the metrics with the gradient of the Neumann solution
        dS = np.einsum('i,ij,kli,kbi,ba,mna->mnjl', self._qwts, L, dL, dL,
                       self._ref_laplacian_pinv, d,
                       optimize=('greedy', sys.maxsize))

        return S + dS

    def _div(self, imass, W):
        # Apply the inverse mass matrix of each element to the weak divergence
        mdiv = imass[:, None] @ W

        # Fold the projection onto the solution basis into the divergence
        div = mdiv @ self.shape.m8 if self.shape.fluxaa else mdiv

        # Order the columns of the divergence as the rows of the flux
        return div.swapaxes(1, 2).reshape(len(div), self.nupts, -1)

    def _operators(self, spts, pn, names):
        smat, djac = self._smats_djacs_qpts(spts)
        mass, S = self._mass(djac), self._stiffness(smat)
        E, M0 = self._fmass, self.shape.m0

        # Form the face mass matrix acting on the normal flux of the interior
        B = np.einsum('ij,jkl,jm->klim', E, pn, M0,
                      optimize=('greedy', sys.maxsize))
        bnd = B.sum(axis=2)

        # Have the stiffness matrices satisfy the discrete divergence theorem
        Sc = self._correct_stiffness(S, bnd)

        # Invert the mass matrix of each element
        imass = np.linalg.inv(mass)

        # Form the divergence and correction operators
        div = self._div(imass, Sc - B)
        corr = imass @ E
        ops = {'div': div, 'corr': corr}

        # Form the gradient operator from the uncorrected stiffness matrices
        if 'grad' in names:
            grad = imass[:, None] @ S
            ops['grad'] = grad.reshape(len(mass), -1, self.nupts)

        return ops

    def operators(self, names):
        n = self.nphyseles
        spts, pn = self.metrics.spts[:, :n], self.metrics.pnorm_fpts[:, :n]

        # Assemble the operators of a chunk of elements at a time
        chunks = [self._operators(s, p, names)
                  for s, p in zip_chunks(self.chunksz, spts, pn, axis=1)]

        return {name: np.concatenate([c[name] for c in chunks])
                for name in names}

    def basis_integrals(self):
        spts = self.metrics.spts[:, :self.nphyseles]
        wts = np.empty((self.nupts, self.nphyseles))

        # Integrate the nodal basis exactly over each element
        for s, w in zip_chunks(self.chunksz, spts, wts, axis=1):
            smat, djac = self._smats_djacs_qpts(s)
            w[:] = self._mass(djac).sum(axis=1).T

        return wts


class PhysicalAdvectionOperators(BaseAdvectionOperators):
    operators_name = 'physical'
    basis_types = ('polynomial', 'rational')
    physop_names = ('div', 'corr')

    def set_backend(self, backend, nonce):
        super().set_backend(backend, nonce)

        # Register the kernel which applies the per-element operators
        self._be.pointwise.register('pyfr.solvers.baseadvec.kernels.physapply')

    def flux_metrics(self, rgn, pts):
        # Leave the fluxes of the curved elements in physical space
        if rgn == 'curved':
            return 'physical', {}
        else:
            return super().flux_metrics(rgn, pts)

    @cached_property
    def _nphyseles(self):
        return self.mesh_regions.get('curved', 0)

    @cached_property
    def _physassembly(self):
        return PhysicalAssembly(self.shape, self.metrics, self._nphyseles)

    def mean_wts_np(self):
        wts = super().mean_wts_np()

        # Integrate the nodal basis exactly over each physical element
        wts[:, :self._nphyseles] = self._physassembly.basis_integrals()

        return wts

    @cached_property
    def _physops(self):
        # Build the per-element operators
        mats = self._physassembly.operators(self.physop_names)

        # Allocate and upload the operators
        return {name: self._be.tiled_matrix(m.shape[1:], len(m), m)
                for name, m in mats.items()}

    def _physapply(self, name, b, out, add=False):
        op = self._physops[name]

        # Apply the operator of each element to its input rows
        tplargs = {
            'nrow': op.nrow, 'ncol': op.ncol, 'nvars': self.nvars, 'add': add
        }

        return self._be.kernel(
            'physapply', tplargs=tplargs, dims=[self._nphyseles], op=op,
            b=b, out=out
        )

    def _make_region_kernel(self, pkern, lkern, b, out, **kwargs):
        s, kerns = self._slice_mat, {'curved': pkern, 'linear': lkern}

        # Apply to each region the operators which act on its elements
        klist = [kerns[rgn](s(b, rgn), s(out, rgn), **kwargs)
                 for rgn in self.mesh_regions]

        return self._make_sliced_kernel(klist)

    def divpcorf(self, f, out):
        pkern, lkern = partial(self._physapply, 'div'), super().divpcorf

        return self._make_region_kernel(pkern, lkern, f, out)

    def divconf(self, fn, out, add):
        pkern, lkern = partial(self._physapply, 'corr'), super().divconf

        return self._make_region_kernel(pkern, lkern, fn, out, add=add)

    @memoize
    def div_metrics(self):
        regions = self.mesh_regions

        if 'linear' not in regions:
            return 'physical', {}
        elif 'curved' not in regions:
            return super().div_metrics()
        else:
            rcpdjac = self.rcpdjac_at_np('upts').copy()

            # Leave the divergence of the physical operators unscaled
            rcpdjac[:, :self._nphyseles] = 1

            rcpdjac = self._be.const_matrix(rcpdjac, tags={'align'})

            return 'transformed', {'rcpdjac': rcpdjac}
