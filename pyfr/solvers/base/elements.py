from dataclasses import dataclass
from functools import cached_property, wraps

import numpy as np

from pyfr.cache import memoize
from pyfr.exprs import npeval
from pyfr.metrics import MeshMetrics
from pyfr.nputil import fuzzysort
from pyfr.quadrules import get_quadrule
from pyfr.shapes import interp_pts, proj_l2


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
    # If the solver computes corrected gradients of the solution
    has_grad_soln = False

    @staticmethod
    def stats_tables(cfg):
        return []

    @staticmethod
    def auxvars(ndims, cfg):
        return {}

    @staticmethod
    def con_is_admissible(cons, cfg):
        # Absent an equation of state every state is admissible
        return np.ones(np.shape(cons[0]), dtype=bool)

    def __init__(self, basiscls, eles, cfg, curved):
        self._be = None

        self.eles = eles
        self.cfg = cfg

        self.nspts = nspts = eles.shape[0]
        self.neles = eles.shape[1]
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

        # Metric terms of the mapping from the reference element
        self.metrics = MeshMetrics(basis, eles, curved)
        self.ncurved = self.metrics.ncurved

        if self.basis.fpts_in_upts:
            self.get_vect_fpts_for_inters = self._get_vect_upts_for_inters
        else:
            self.get_vect_fpts_for_inters = self._get_vect_fpts_for_inters

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

        # Interpolate its solution onto our solution points
        interp = solnb.ubasis.nodal_basis_at(self.basis.upts)
        return interp_pts(interp, solnmat[:, :self.nvars])

    @cached_property
    def plocfpts(self):
        # Interpolate the physical mesh locations to the flux points
        plocop = self.basis.sbasis.nodal_basis_at(self.basis.fpts)
        return interp_pts(plocop, self.eles)

    @cached_property
    def _scal_upts_tmp(self):
        return self._be.matrix((self.nupts, self.nvars, self.neles),
                               tags={'align'})

    @cached_property
    def srtd_face_fpts(self):
        plocfpts = self.plocfpts
        sffpts = []

        for ffpts in self.basis.facefpts:
            ffpts = np.asarray(ffpts)
            coords = plocfpts[ffpts].transpose(1, 2, 0)
            perm = fuzzysort(coords)
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

    def set_backend(self, backend, nonce):
        self._be = backend

        # If we are doing gradient fusion
        self.grad_fusion = not (self._be.blocks or 'flux' in self.antialias)

        if self.basis.order >= 2:
            self.linoff = self.ncurved - self.ncurved % -backend.csubsz
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

        if 'grad_upts' in sbufs and self.grad_fusion:
            self._grad_upts = valloc('vect_fpts', nupts)
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

    def jac_at_np(self, name):
        return self.metrics.jac_at(name)

    def smat_at_np(self, name):
        return self.metrics.smat_at(name)

    @memoize
    def curved_smat_at(self, name):
        smat = self.smat_at_np(name)[..., :self.linoff]
        return self._be.const_matrix(smat, tags={'align'})

    def rcpdjac_at_np(self, name):
        return self.metrics.rcpdjac_at(name)

    @sliceat
    @memoize
    def rcpdjac_at(self, name):
        return self._be.const_matrix(self.rcpdjac_at_np(name), tags={'align'})

    @cached_property
    def mean_wts(self):
        wts = self.basis.ubasis.invvdm[:, 0, None]/self.rcpdjac_at_np('upts')
        wts /= np.sum(wts, axis=0)

        return self._be.const_matrix(wts, tags={'align'})

    def ploc_at_np(self, name):
        return self.metrics.ploc_at(name)

    @sliceat
    @memoize
    def ploc_at(self, name):
        return self._be.const_matrix(self.ploc_at_np(name), tags={'align'})

    @memoize
    def vtx_interp_at(self, name):
        op = self.basis.lbasis.nodal_basis_at(getattr(self.basis, name))
        return self._be.const_matrix(op, tags={'align'})

    @cached_property
    def upts(self):
        return self._be.const_matrix(self.basis.upts)

    @cached_property
    def qpts(self):
        return self._be.const_matrix(self.basis.qpts)

    @cached_property
    def _pnorm_fpts(self):
        return np.vstack([self.pnorm_at(self.basis.fpts[fp], fidx)
                          for fidx, fp in enumerate(self.basis.facefpts)])

    def pnorm_at(self, name, fidx):
        return self.metrics.pnorm_at(name, fidx)

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
    def get_comm_fpts_for_inters(self, eidxs, fidx):
        return self._comm_fpts.mid, self.srtd_face_fpts[fidx][eidxs]

    def get_ploc_for_inters(self, eidxs, fidx):
        fpts_idx = self.srtd_face_fpts[fidx][eidxs]
        ploc = self.plocfpts[fpts_idx, eidxs[:, None]]
        return ploc.reshape(-1, self.ndims),
