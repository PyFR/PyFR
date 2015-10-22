# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from pyfr.nputil import npeval, fuzzysort
from pyfr.util import memoize


class BaseElements(object, metaclass=ABCMeta):
    privarmap = None
    convarmap = None

    def __init__(self, basiscls, eles, cfg):
        self._be = None

        self.eles = eles
        self.cfg = cfg

        self.nspts = nspts = eles.shape[0]
        self.neles = neles = eles.shape[1]
        self.ndims = ndims = eles.shape[2]

        # Kernels we provide
        self.kernels = {}

        # Check the dimensionality of the problem
        if ndims != basiscls.ndims or ndims not in self.privarmap:
            raise ValueError('Invalid element matrix dimensions')

        # Determine the number of dynamical variables
        self.nvars = len(self.privarmap[ndims])

        # Instantiate the basis class
        self.basis = basis = basiscls(nspts, cfg)

        # See what kind of projection the basis is using
        self.antialias = basis.antialias

        # If we need quadrature points or not
        haveqpts = 'flux' in self.antialias or 'div-flux' in self.antialias

        # Sizes
        self.nupts = basis.nupts
        self.nqpts = basis.nqpts if haveqpts else None
        self.nfpts = basis.nfpts
        self.nfacefpts = basis.nfacefpts

        # Construct smat_base for free-stream metric
        self.metric = self.basis.metric
        if self.metric == 'free-stream':
            self._smat_base()

        # Physical normals at the flux points
        self._gen_pnorm_fpts()

        # Construct the physical location operator matrix
        plocop = basis.sbasis.nodal_basis_at(basis.fpts)

        # Apply the operator to the mesh elements and reshape
        plocfpts = np.dot(plocop, eles.reshape(nspts, -1))
        self.plocfpts = plocfpts.reshape(self.nfpts, neles, ndims)
        plocfpts = self.plocfpts.transpose(1, 2, 0).tolist()

        self._srtd_face_fpts = [[fuzzysort(pts, ffpts) for pts in plocfpts]
                                for ffpts in basis.facefpts]

    @abstractmethod
    def pri_to_conv(ics, cfg):
        pass

    def set_ics_from_cfg(self):
        # Bring simulation constants into scope
        vars = self.cfg.items_as('constants', float)

        if any(d in vars for d in 'xyz'):
            raise ValueError('Invalid constants (x, y, or z) in config file')

        # Get the physical location of each solution point
        coords = self.ploc_at_np('upts').swapaxes(0, 1)
        vars.update(dict(zip('xyz', coords)))

        # Evaluate the ICs from the config file
        ics = [npeval(self.cfg.get('soln-ics', dv), vars)
               for dv in self.privarmap[self.ndims]]

        # Allocate
        self._scal_upts = np.empty((self.nupts, self.nvars, self.neles))

        # Convert from primitive to conservative form
        for i, v in enumerate(self.pri_to_conv(ics, self.cfg)):
            self._scal_upts[:, i, :] = v

    def set_ics_from_soln(self, solnmat, solncfg):
        # Recreate the existing solution basis
        solnb = self.basis.__class__(None, solncfg)

        # Form the interpolation operator
        interp = solnb.ubasis.nodal_basis_at(self.basis.upts)

        # Sizes
        nupts, neles, nvars = self.nupts, self.neles, self.nvars

        # Apply and reshape
        self._scal_upts = np.dot(interp, solnmat.reshape(solnb.nupts, -1))
        self._scal_upts = self._scal_upts.reshape(nupts, nvars, neles)

    @abstractproperty
    def _scratch_bufs(self):
        pass

    @abstractmethod
    def set_backend(self, backend, nscal_upts):
        self._be = backend

        # Sizes
        ndims, nvars, neles = self.ndims, self.nvars, self.neles
        nfpts, nupts, nqpts = self.nfpts, self.nupts, self.nqpts
        sbufs, abufs = self._scratch_bufs, []

        # Convenience functions for scalar/vector allocation
        alloc = lambda ex, n: abufs.append(
            backend.matrix(n, extent=ex, tags={'align'})
        ) or abufs[-1]
        salloc = lambda ex, n: alloc(ex, (n, nvars, neles))
        valloc = lambda ex, n: alloc(ex, (ndims, n, nvars, neles))

        # Allocate required scalar scratch space
        if 'scal_fpts' in sbufs and 'scal_qpts' in sbufs:
            self._scal_fqpts = salloc('_scal_fqpts', nfpts + nqpts)
            self._scal_fpts = self._scal_fqpts.rslice(0, nfpts)
            self._scal_qpts = self._scal_fqpts.rslice(nfpts, nfpts + nqpts)
        elif 'scal_fpts' in sbufs:
            self._scal_fpts = salloc('scal_fpts', nfpts)
        elif 'scal_qpts' in sbufs:
            self._scal_qpts = salloc('scal_qpts', nqpts)

        # Allocate required vector scratch space
        if 'vect_upts' in sbufs:
            self._vect_upts = valloc('vect_upts', nupts)
        if 'vect_qpts' in sbufs:
            self._vect_qpts = valloc('vect_qpts', nqpts)
        if 'vect_fpts' in sbufs:
            self._vect_fpts = valloc('vect_fpts', nfpts)

        # Allocate and bank the storage required by the time integrator
        self._scal_upts = [backend.matrix(self._scal_upts.shape,
                                          self._scal_upts, tags={'align'})
                           for i in range(nscal_upts)]
        self.scal_upts_inb = inb = backend.matrix_bank(self._scal_upts)
        self.scal_upts_outb = backend.matrix_bank(self._scal_upts)

        # Find/allocate space for a solution-sized scalar that is
        # allowed to alias other scratch space in the simulation
        aliases = next((m for m in abufs if m.nbytes >= inb.nbytes), None)
        self._scal_upts_temp = backend.matrix(inb.ioshape, aliases=aliases,
                                              tags=inb.tags)

    @memoize
    def opmat(self, expr):
        return self._be.const_matrix(self.basis.opmat(expr),
                                     tags={expr, 'align'})

    @memoize
    def smat_at_np(self, name):
        return self._get_smats(getattr(self.basis, name))

    @memoize
    def smat_at(self, name):
        return self._be.const_matrix(self.smat_at_np(name), tags={'align'})

    @memoize
    def rcpdjac_at_np(self, name):
        _, djac = self._get_smats(getattr(self.basis, name), retdets=True)

        if np.any(djac < -1e-5):
            raise RuntimeError('Negative mesh Jacobians detected')

        return 1.0 / djac

    @memoize
    def rcpdjac_at(self, name):
        return self._be.const_matrix(self.rcpdjac_at_np(name), tags={'align'})

    @memoize
    def ploc_at_np(self, name):
        op = self.basis.sbasis.nodal_basis_at(getattr(self.basis, name))

        ploc = np.dot(op, self.eles.reshape(self.nspts, -1))
        ploc = ploc.reshape(-1, self.neles, self.ndims).swapaxes(1, 2)

        return ploc

    @memoize
    def ploc_at(self, name):
        return self._be.const_matrix(self.ploc_at_np(name), tags={'align'})

    def _gen_pnorm_fpts(self):
        smats = self._get_smats(self.basis.fpts).transpose(1, 3, 0, 2)

        # We need to compute |J|*[(J^{-1})^{T}.N] where J is the
        # Jacobian and N is the normal for each fpt.  Using
        # J^{-1} = S/|J| where S are the smats, we have S^{T}.N.
        pnorm_fpts = np.einsum('ijlk,il->ijk', smats, self.basis.norm_fpts)

        # Compute the magnitudes of these flux point normals
        mag_pnorm_fpts = np.einsum('...i,...i', pnorm_fpts, pnorm_fpts)
        mag_pnorm_fpts = np.sqrt(mag_pnorm_fpts)

        # Check that none of these magnitudes are zero
        if np.any(mag_pnorm_fpts < 1e-10):
            raise RuntimeError('Zero face normals detected')

        # Normalize the physical normals at the flux points
        self._norm_pnorm_fpts = pnorm_fpts / mag_pnorm_fpts[..., None]
        self._mag_pnorm_fpts = mag_pnorm_fpts

    def _get_jac_eles_at(self, pts):
        nspts, neles, ndims = self.nspts, self.neles, self.ndims
        npts = len(pts)

        # Form the Jacobian operator
        jacop = np.rollaxis(self.basis.sbasis.jac_nodal_basis_at(pts), 2)

        # Cast as a matrix multiply and apply to eles
        jac = np.dot(jacop.reshape(-1, nspts), self.eles.reshape(nspts, -1))

        # Reshape (npts*ndims, neles*ndims) => (npts, ndims, neles, ndims)
        jac = jac.reshape(npts, ndims, neles, ndims)

        # Transpose to get (ndims, npts, ndims, neles)
        return jac.transpose(3, 0, 1, 2)

    def _get_smats(self, pts, retdets=False):
        if self.metric == 'free-stream':
            npts = len(pts)
            smats = np.empty((self.ndims, npts, self.ndims*self.neles))

            M0 = self.basis.mbasis.nodal_basis_at(pts)

            for i in range(self.ndims):
                smats[i] = np.dot(M0, self._smats[i])

            smats = smats.reshape(self.ndims, npts, self.ndims, -1)

            if retdets:
                djacs = np.dot(M0, self._djacs)
        else:
            jac = self._get_jac_eles_at(pts)
            smats = np.empty_like(jac)

            if self.ndims == 2:
                a, b, c, d = jac[0,:,0], jac[0,:,1], jac[1,:,0], jac[1,:,1]

                smats[0,:,0], smats[0,:,1] = d, -b
                smats[1,:,0], smats[1,:,1] = -c, a

                if retdets:
                    djacs = a*d - b*c
            else:
                # We note that J = [x0, x1, x2]
                x0, x1, x2 = jac[:,:,0], jac[:,:,1], jac[:,:,2]

                smats[0] = np.cross(x1, x2, axisa=0, axisb=0, axisc=1)
                smats[1] = np.cross(x2, x0, axisa=0, axisb=0, axisc=1)
                smats[2] = np.cross(x0, x1, axisa=0, axisb=0, axisc=1)

                if retdets:
                    # Exploit the fact that det(J) = x0 . (x1 ^ x2)
                    djacs = np.einsum('ij...,ji...->j...', x0, smats[0])

        return (smats, djacs) if retdets else smats

    def _plot_at_pts_np(self, pts):
        op = self.basis.sbasis.nodal_basis_at(pts)

        ploc = np.dot(op, self.eles.reshape(self.nspts, -1))
        ploc = ploc.reshape(-1, self.neles, self.ndims).swapaxes(1, 2)

        return ploc

    def _smat_base(self):
        # Metric basis
        mpts = self.basis.mpts
        mbasis = self.basis.mbasis
        self.nmpts = nmpts = len(mpts)

        # Dimensions and number of elements
        ndims = self.ndims
        neles = self.neles

        # Coordinate at pts
        x = self._plot_at_pts_np(mpts)

        # Jacobian at pts
        jacop = np.rollaxis(mbasis.jac_nodal_basis_at(mpts), 2)
        jacop = jacop.reshape(-1, nmpts)

        # Cast as a matrix multiply and apply to eles
        jac = np.dot(jacop, x.reshape(nmpts, -1))

        # Reshape (npts*ndims, neles*ndims) => (npts, ndims, neles, ndims)
        jac = jac.reshape(nmpts, ndims, ndims, neles)

        # Transpose to get (ndims, npts, ndims, neles)
        jac = jac.transpose(2, 0, 1, 3)

        smats = np.empty_like(jac)

        if ndims == 2:
            a, b, c, d = jac[0,:,0], jac[0,:,1], jac[1,:,0], jac[1,:,1]

            smats[0,:,0], smats[0,:,1] = d, -b
            smats[1,:,0], smats[1,:,1] = -c, a

            self._djacs = a*d - b*c
        else:
            # We note that J = [x0, x1, x2]
            x0, x1, x2 = jac[:,:,0], jac[:,:,1], jac[:,:,2]

            # Cpmpute x cross x_(chi)
            tt = np.zeros((ndims,) + x.shape)
            tt[0] = np.cross(x0, x, axisa=0, axisb=1, axisc=1)
            tt[1] = np.cross(x1, x, axisa=0, axisb=1, axisc=1)
            tt[2] = np.cross(x2, x, axisa=0, axisb=1, axisc=1)
            tt = tt.reshape(ndims, nmpts, -1)

            # Derivative
            dtt = np.zeros((ndims, nmpts*ndims, ndims*neles))
            dtt[0] = np.dot(jacop, tt[0])
            dtt[1] = np.dot(jacop, tt[1])
            dtt[2] = np.dot(jacop, tt[2])
            dtt = dtt.reshape(ndims, nmpts, ndims, ndims, -1).swapaxes(1, 2)

            smats[0] = 0.5*(dtt[1][2] - dtt[2][1])
            smats[1] = 0.5*(dtt[2][0] - dtt[0][2])
            smats[2] = 0.5*(dtt[0][1] - dtt[1][0])

            # Exploit the fact that det(J) = x0 . (x1 ^ x2)
            self._djacs = np.einsum('ij...,ji...->j...', x0, smats[0])

        # Reshape
        self._smats = smats.reshape(ndims, self.nmpts, -1)

    def get_mag_pnorms(self, eidx, fidx):
        fpts_idx = self.basis.facefpts[fidx]
        return self._mag_pnorm_fpts[fpts_idx,eidx]

    def get_mag_pnorms_for_inter(self, eidx, fidx):
        fpts_idx = self._srtd_face_fpts[fidx][eidx]
        return self._mag_pnorm_fpts[fpts_idx,eidx]

    def get_norm_pnorms_for_inter(self, eidx, fidx):
        fpts_idx = self._srtd_face_fpts[fidx][eidx]
        return self._norm_pnorm_fpts[fpts_idx,eidx]

    def get_norm_pnorms(self, eidx, fidx):
        fpts_idx = self.basis.facefpts[fidx]
        return self._norm_pnorm_fpts[fpts_idx,eidx]

    def get_scal_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[fidx][eidx]]
        cstri = ((self._scal_fpts.leadsubdim,),)*nfp

        return (self._scal_fpts.mid,)*nfp, rcmap, cstri

    def get_vect_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[fidx][eidx]]
        rcstri = ((self.nfpts, self._vect_fpts.leadsubdim),)*nfp

        return (self._vect_fpts.mid,)*nfp, rcmap, rcstri

    def get_avis_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[fidx][eidx]]
        cstri = ((self._avis_fpts.leadsubdim,),)*nfp

        return (self._avis_fpts.mid,)*nfp, rcmap, cstri

    def get_ploc_for_inter(self, eidx, fidx):
        fpts_idx = self._srtd_face_fpts[fidx][eidx]
        return self.plocfpts[fpts_idx,eidx]
