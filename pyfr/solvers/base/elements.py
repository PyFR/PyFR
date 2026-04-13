from dataclasses import dataclass
from functools import cached_property, wraps

import numpy as np

from pyfr.cache import memoize
from pyfr.nputil import batched_fuzzysort, npeval
from pyfr.quadrules import get_quadrule
from pyfr.shapes import proj_l2


@dataclass
class ExportableField:
    name: str
    shape: tuple
    getter: callable
    dtype: type = None


def inters_map(meth):
    @wraps(meth)
    def newmeth(self, eidxs, fidx):
        nfp = self.nfacefpts[fidx]
        n = len(eidxs)
        cmap = np.repeat(eidxs, nfp)

        match meth(self, eidxs, fidx):
            case [mid, rmap]:
                return np.full(n*nfp, mid), rmap.ravel(), cmap
            case [mid, rmap, lda]:
                return (np.full(n*nfp, mid), rmap.ravel(),
                        cmap, np.full(n*nfp, lda))
    return newmeth


class BaseElements:
    def __init__(self, basiscls, eles, cfg):
        self._be = None

        self.eles = eles
        self.cfg = cfg

        self.nspts = nspts = eles.shape[0]
        self.neles = neles = eles.shape[1]
        self.ndims = ndims = eles.shape[2]

        # Field variables
        self.convars = self.convars(ndims, cfg)
        self.privars = self.privars(ndims, cfg)
        self.dualcoeffs = self.dualcoeffs(ndims, cfg)

        # Kernels we provide
        self.kernels = {}

        # Exportable fields
        self.export_fields = []

        # Check the dimensionality of the problem
        if ndims != basiscls.ndims:
            raise ValueError('Invalid element matrix dimensions')

        # Determine the number of dynamical variables
        self.nvars = len(self.convars)

        # Instantiate the basis class
        self.basis = basis = basiscls(nspts, cfg)
        self.name = basis.name

        # See what kind of projection the basis is using
        self.antialias = basis.antialias

        # Sizes
        self.nupts = basis.nupts
        self.nqpts = basis.nqpts if 'flux' in self.antialias else None
        self.nfpts = basis.nfpts
        self.nfacefpts = basis.nfacefpts
        self.nmpts = basis.nmpts

        if self.basis.fpts_in_upts:
            self.get_vect_fpts_for_inters = self._get_vect_upts_for_inters
            self.get_comm_fpts_for_inters = self._get_comm_fpts_for_inters
        else:
            self.get_vect_fpts_for_inters = self._get_vect_fpts_for_inters
            self.get_comm_fpts_for_inters = self._get_vect_fpts_for_inters

    def set_ics_from_cfg(self):
        # Bring simulation constants into scope
        vars = self.cfg.items_as('constants', float)

        if any(d in vars for d in 'xyz'):
            raise ValueError('Invalid constants (x, y, or z) in config file')

        # See if performing L2 projection
        ename = self.basis.name
        upts = self.cfg.get(f'solver-elements-{ename}', 'soln-pts')
        qdeg = (self.cfg.getint('soln-ics', f'quad-deg-{ename}', 0) or
                self.cfg.getint('soln-ics', 'quad-deg', 0))

        # Default to solution points if quad-pts are not specified
        qpts = self.cfg.get('soln-ics', f'quad-pts-{ename}', upts)

        # Get the physical location of each interpolation point
        if qdeg:
            qrule = get_quadrule(ename, qpts, qdeg=qdeg)
            coords = self.ploc_at_np(qrule.pts)

            # Compute projection operator
            m8 = proj_l2(qrule, self.basis.ubasis)
        else:
            m8 = None
            coords = self.ploc_at_np('upts')

        vars |= dict(zip('xyz', coords.swapaxes(0, 1)))

        # Evaluate the ICs from the config file
        ics = [npeval(self.cfg.getexpr('soln-ics', dv), vars)
               for dv in self.privars]

        # Allocate
        scal_upts = np.empty((self.nupts, self.nvars, self.neles))

        # Convert from primitive to conservative form
        for i, v in enumerate(self.pri_to_con(ics, self.cfg)):
            if m8 is not None:
                v = m8 @ np.broadcast_to(v, (m8.shape[1], self.neles))

            scal_upts[:, i] = v

        return scal_upts

    def set_ics_from_soln(self, solnmat, solncfg):
        # Recreate the existing solution basis
        solnb = self.basis.__class__(None, solncfg)

        # Form the interpolation operator
        interp = solnb.ubasis.nodal_basis_at(self.basis.upts)

        # Sizes
        nupts, neles, nvars = self.nupts, self.neles, self.nvars

        # Apply and reshape
        scal_upts = interp @ solnmat.reshape(solnb.nupts, -1)
        return scal_upts.reshape(nupts, nvars, neles)

    @cached_property
    def plocfpts(self):
        # Construct the physical location operator matrix
        plocop = self.basis.sbasis.nodal_basis_at(self.basis.fpts)

        # Apply the operator to the mesh elements and reshape
        plocfpts = plocop @ self.eles.reshape(self.nspts, -1)
        plocfpts = plocfpts.reshape(self.nfpts, self.neles, self.ndims)

        return plocfpts

    @cached_property
    def _scal_upts_cpy(self):
        return self._be.matrix((self.nupts, self.nvars, self.neles),
                               tags={'align'})

    @cached_property
    def srtd_face_fpts(self):
        plocfpts = self.plocfpts
        sffpts = []

        for ffpts in self.basis.facefpts:
            ffpts = np.asarray(ffpts)
            coords = plocfpts[ffpts].transpose(1, 2, 0)
            perm = batched_fuzzysort(coords)
            sffpts.append(ffpts[perm])

        return sffpts

    def _scratch_bufs(self):
        pass

    @property
    def mesh_regions(self):
        off = self.linoff

        # No curved elements
        if off == 0:
            return {'linear': self.neles}
        # All curved elements
        elif off >= self.neles:
            return {'curved': self.neles}
        # Mix of curved and linear elements
        else:
            return {'curved': off, 'linear': self.neles - off}

    def _slice_mat(self, mat, region, ra=None, rb=None):
        if mat is None:
            return None

        off = self.linoff

        # Handle stacked matrices
        if len(mat.ioshape) >= 3:
            off *= mat.ioshape[-2]
        else:
            off = min(off, mat.ncol)

        if region == 'curved':
            return mat.slice(ra, rb, 0, off)
        elif region == 'linear':
            return mat.slice(ra, rb, off, mat.ncol)
        else:
            raise ValueError('Invalid slice region')

    def _make_sliced_kernel(self, kseq):
        klist = list(kseq)

        if len(klist) > 1:
            return self._be.unordered_meta_kernel(klist, [self.linoff])
        else:
            return klist[0]

    def set_backend(self, backend, nonce, linoff):
        self._be = backend

        # If we are doing gradient fusion
        self.grad_fusion = not (self._be.blocks or 'flux' in self.antialias)

        if self.basis.order >= 2:
            self.linoff = linoff - linoff % -backend.csubsz
        else:
            self.linoff = self.neles

        # Sizes
        ndims, nvars, neles = self.ndims, self.nvars, self.neles
        nfpts, nupts, nqpts = self.nfpts, self.nupts, self.nqpts
        sbufs, abufs = self._scratch_bufs, []

        # Convenience functions for scalar/vector allocation
        alloc = lambda ex, n: abufs.append(
            backend.matrix(n, extent=nonce + ex, tags={'align'})
        ) or abufs[-1]
        salloc = lambda ex, n: alloc(ex, (n, nvars, neles))
        valloc = lambda ex, n: alloc(ex, (ndims, n, nvars, neles))

        # Allocate required scalar scratch space
        if 'scal_fpts' in sbufs:
            self._scal_fpts = salloc('scal_fpts', nfpts)
        if 'scal_qpts' in sbufs:
            self._scal_qpts = salloc('scal_qpts', nqpts)

        # Allocate required vector scratch space
        if 'vect_upts' in sbufs:
            self._vect_upts = valloc('vect_upts', nupts)
        if 'vect_qpts' in sbufs:
            self._vect_qpts = valloc('vect_qpts', nqpts)
        if 'vect_fpts' in sbufs:
            self._vect_fpts = valloc('vect_fpts', nfpts)

        # Allocate space if needed for interfaces
        if 'comm_fpts' in sbufs:
            self._comm_fpts = salloc('comm_fpts', nfpts)
        elif 'vect_fpts' in sbufs:
            self._comm_fpts = self._vect_fpts.slice(0, self.nfpts)

        if 'grad_upts' in sbufs and self.grad_fusion:
            self._grad_upts = valloc('grad_upts', nupts)
        elif hasattr(self, '_vect_upts'):
            self._grad_upts = self._vect_upts

        self.scal_upts = []

    def alloc_bank(self, extent, ic=None):
        shape = (self.nupts, self.nvars, self.neles)
        m = self._be.matrix(shape, ic, extent=extent, tags={'align'})
        self.scal_upts.append(m)
        return m

    @memoize
    def opmat(self, expr):
        return self._be.const_matrix(self.basis.opmat(expr),
                                     tags={expr, 'align'})

    def sliceat(fn):
        @memoize
        @wraps(fn)
        def newfn(self, name, side=None):
            mat = fn(self, name)

            if side is not None:
                return self._slice_mat(mat, side)
            else:
                return mat

        return newfn

    @memoize
    def smat_at_np(self, name):
        smats_mpts, _ = self._smats_djacs_mpts

        # Interpolation matrix to pts
        pt = getattr(self.basis, name) if isinstance(name, str) else name
        m0 = self.basis.mbasis.nodal_basis_at(pt)

        # Interpolate the smats
        smats = np.array([m0 @ smat for smat in smats_mpts])
        return smats.reshape(self.ndims, -1, self.ndims, self.neles)

    @memoize
    def curved_smat_at(self, name):
        smat = self.smat_at_np(name)[..., :self.linoff]
        return self._be.const_matrix(smat, tags={'align'})

    @memoize
    def rcpdjac_at_np(self, name):
        _, djacs_mpts = self._smats_djacs_mpts

        # Interpolation matrix to pts
        pt = getattr(self.basis, name) if isinstance(name, str) else name
        m0 = self.basis.mbasis.nodal_basis_at(pt)

        # Interpolate the djacs
        djac = m0 @ djacs_mpts

        if np.any(djac < -1e-5):
            raise RuntimeError('Negative mesh Jacobians detected')

        return 1.0 / djac

    @sliceat
    @memoize
    def rcpdjac_at(self, name):
        return self._be.const_matrix(self.rcpdjac_at_np(name), tags={'align'})

    @cached_property
    def mean_wts(self):
        wts = self.basis.ubasis.invvdm[:, 0, None]/self.rcpdjac_at_np('upts')
        wts /= np.sum(wts, axis=0)

        return self._be.const_matrix(wts, tags={'align'})

    @memoize
    def ploc_at_np(self, name):
        pt = getattr(self.basis, name) if isinstance(name, str) else name
        op = self.basis.sbasis.nodal_basis_at(pt)

        ploc = op @ self.eles.reshape(self.nspts, -1)
        ploc = ploc.reshape(len(pt), -1, self.ndims).swapaxes(1, 2)

        return ploc

    @sliceat
    @memoize
    def ploc_at(self, name):
        return self._be.const_matrix(self.ploc_at_np(name), tags={'align'})

    @cached_property
    def upts(self):
        return self._be.const_matrix(self.basis.upts)

    @cached_property
    def qpts(self):
        return self._be.const_matrix(self.basis.qpts)

    @cached_property
    def _pnorm_fpts(self):
        return self.pnorm_at('fpts', self.basis.norm_fpts)

    @memoize
    def pnorm_at(self, name, norm):
        smats = self.smat_at_np(name).transpose(1, 3, 0, 2)

        # We need to compute |J|*[(J^{-1})^{T}.N] where J is the
        # Jacobian and N is the normal for each point. Using
        # J^{-1} = S/|J| where S are the smats, we have S^{T}.N.
        pnorm = np.einsum('ijlk,il->ijk', smats, norm)

        # Compute the magnitudes of these flux point normals
        mag_pnorm = np.einsum('...i,...i', pnorm, pnorm)

        # Check that none of these magnitudes are zero
        if np.any(np.sqrt(mag_pnorm) < 1e-10):
            raise RuntimeError('Zero face normals detected')

        return pnorm

    @cached_property
    def _smats_djacs_mpts(self):
        # Metric basis with grid point (q<=p) or pseudo grid points (q>p)
        mpts = self.basis.mpts
        mbasis = self.basis.mbasis

        # Dimensions, number of elements and number of mpts
        ndims, neles, nmpts = self.ndims, self.neles, self.nmpts

        # Physical locations of the pseudo grid points
        x = self.ploc_at_np('mpts')

        # Jacobian operator at these points
        jacop = np.rollaxis(mbasis.jac_nodal_basis_at(mpts), 2)
        jacop = jacop.reshape(-1, nmpts)

        # Cast as a matrix multiply and apply to eles
        jac = jacop @ x.reshape(nmpts, -1)

        # Reshape (nmpts*ndims, neles*ndims) => (nmpts, ndims, neles, ndims)
        jac = jac.reshape(nmpts, ndims, ndims, neles)

        # Transpose to get (ndims, ndims, nmpts, neles)
        jac = jac.transpose(1, 2, 0, 3)

        smats = np.empty((ndims, nmpts, ndims, neles))

        if ndims == 2:
            a, b, c, d = jac[0, 0], jac[1, 0], jac[0, 1], jac[1, 1]

            smats[0, :, 0], smats[0, :, 1] = d, -b
            smats[1, :, 0], smats[1, :, 1] = -c, a

            djacs = a*d - b*c
        else:
            dtt = []
            for dx in jac:
                # Compute x cross x_(chi)
                tt = np.cross(x, dx, axisa=1, axisb=0, axisc=1)

                # Jacobian of x cross x_(chi) at the pseudo grid points
                dt = jacop @ tt.reshape(nmpts, -1)
                dt = dt.reshape(nmpts, ndims, ndims, -1).swapaxes(0, 1)

                dtt.append(dt)

            # Kopriva's invariant form of smats; JSC 26(3), 301-327, Eq. (37)
            smats[0] = 0.5*(dtt[2][1] - dtt[1][2])
            smats[1] = 0.5*(dtt[0][2] - dtt[2][0])
            smats[2] = 0.5*(dtt[1][0] - dtt[0][1])

            # We note that J = [x0; x1; x2]
            x0, x1, x2 = jac

            # Exploit the fact that det(J) = x0 · (x1 ^ x2)
            x1cx2 = np.cross(x1, x2, axisa=0, axisb=0, axisc=1)
            djacs = np.einsum('ij...,ji...->j...', x0, x1cx2)

        return smats.reshape(ndims, nmpts, -1), djacs

    def get_pnorms(self, eidx, fidx):
        fpts_idx = self.basis.facefpts[fidx]
        return self._pnorm_fpts[fpts_idx, eidx]

    def get_pnorms_for_inters(self, eidxs, fidx):
        fpts_idx = self.srtd_face_fpts[fidx][eidxs]
        pn = self._pnorm_fpts[fpts_idx, eidxs[:, None]]
        return pn.reshape(-1, self.ndims),

    @inters_map
    def _get_vect_fpts_for_inters(self, eidxs, fidx):
        rmap = self.srtd_face_fpts[fidx][eidxs]
        return self._vect_fpts.mid, rmap, self.nfpts

    @inters_map
    def _get_vect_upts_for_inters(self, eidxs, fidx):
        rmap = self.srtd_face_fpts[fidx][eidxs]
        fmap = self.basis.fpts_map_upts[rmap]
        return self._vect_upts.mid, fmap, self.nupts

    @inters_map
    def _get_comm_fpts_for_inters(self, eidxs, fidx):
        return self._comm_fpts.mid, self.srtd_face_fpts[fidx][eidxs]

    def get_ploc_for_inters(self, eidxs, fidx):
        fpts_idx = self.srtd_face_fpts[fidx][eidxs]
        ploc = self.plocfpts[fpts_idx, eidxs[:, None]]
        return ploc.reshape(-1, self.ndims),
