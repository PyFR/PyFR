import itertools as it
from math import comb

from boostree import RTree
import numpy as np

from pyfr.mpiutil import (AlltoallMixin, SharedWorkQueue, Sorter,
                          get_comm_rank_root, get_node_comm, mpi,
                          mpi_progress, shared_ndarrays)
from pyfr.nputil import morton_encode, zip_chunks
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


def get_interpolator(name, ndims, is_admissible, opts, order=None):
    cls = subclass_where(BaseInterpolator, name=name)

    return cls.from_opts(ndims, is_admissible, opts, order=order)


def _ploc_op(etype, nspts, cfg):
    # Shape and associated basis
    shapecls = subclass_where(BaseShape, name=etype)
    basis = shapecls(nspts, cfg)

    # Interpolate the shape points to the solution points
    return basis.sbasis.nodal_basis_at(basis.upts)


class BaseInterpolator:
    name = None

    int_opts = {}
    float_opts = {}
    enum_opts = {}
    dflt_opts = {}

    @classmethod
    def from_opts(cls, ndims, is_admissible, opts={}, *, order=None):
        # Normalise hyphens to underscores in user-supplied keys
        opts = {k.replace('-', '_'): v for k, v in opts.items()}

        kwargs = {}
        for k, v in dict(cls.dflt_opts, **opts).items():
            if k in cls.int_opts:
                kwargs[k] = int(v)
            elif k in cls.float_opts:
                kwargs[k] = float(v)
            elif k in cls.enum_opts:
                kwargs[k] = cls.enum_opts[k][v]
            else:
                raise ValueError('Invalid option')

        if order is not None:
            kwargs['order'] = order

        return cls(ndims, is_admissible, **kwargs)


class IDWInterpolator(BaseInterpolator):
    name = 'idw'

    # Integer options
    int_opts = {'n'}

    # Floating point options
    float_opts = {'rho'}

    # Default options
    dflt_opts = {'n': 0, 'rho': 0}

    def __init__(self, ndims, is_admissible, *, order=None, n=0, rho=0):
        self.is_admissible = is_admissible
        self.n = n or 2**ndims
        self.rho = rho or ndims + 1
        self.eps = 10*np.finfo(float).eps

    def __call__(self, pts, spts, svals):
        dists = np.linalg.norm(spts - pts[:, None], axis=2)

        swts = np.maximum(dists, self.eps)**-self.rho
        swts /= swts.sum(axis=1, keepdims=True)
        out = (swts[:, None] @ svals)[:, 0]

        # Exact hits take the value of the coincident source point
        amin = dists.argmin(axis=1)
        dmin = np.take_along_axis(dists, amin[:, None], 1)[:, 0]
        if (ex := dmin < self.eps).any():
            out[ex] = svals[ex, amin[ex]]

        return out


class WENOInterpolator(BaseInterpolator):
    name = 'weno'

    int_opts = {'n', 'degree', 'sub_degree', 'nsub', 'sub_n'}
    float_opts = {'rho', 'q', 'ct', 'cond', 'dir_bias', 'gamma0'}
    enum_opts = {'mode': {'wenoz': 'wenoz', 'teno': 'teno'}}
    dflt_opts = {'n': 0, 'degree': 0, 'sub_degree': 0, 'nsub': 0,
                 'sub_n': 0, 'rho': 0, 'q': 4, 'ct': 1.0e-3,
                 'cond': 1.0e8, 'dir_bias': 2.5, 'gamma0': 0.85,
                 'mode': 'wenoz'}

    def __init__(self, ndims, is_admissible, *, order=None, n=0, degree=0,
                 sub_degree=0, nsub=0, sub_n=0, rho=0, q=4, ct=1.0e-3,
                 cond=1.0e8, dir_bias=2.5, gamma0=0.85, mode='wenoz'):
        # Auto-derive degree from the source polynomial order
        if not degree:
            degree = min(order, 3) if order else 2

        if not sub_degree:
            sub_degree = max(degree - 1, 1)

        # Auto-derive the stencil size to be ~3x overdetermined
        nterms = comb(ndims + degree, degree)
        if not n:
            n = max(3*nterms, 24 if ndims == 2 else 48)

        self.is_admissible = is_admissible
        self.n = n
        self.q = q
        self.ct = ct
        self.cond = cond
        self.dir_bias = dir_bias
        self.gamma0 = gamma0
        self.eps = 10*np.finfo(float).eps

        # Gram ridge and Cholesky screening threshold; ridge << screen**2
        self.ridge = 1.0e-10
        self.screen = 1.0e-2

        # Keep an IDW fallback for exact hits and rejected stencils
        self._idw = IDWInterpolator(ndims, is_admissible, n=self.n,
                                    rho=rho or ndims + 1)

        # Precompute the polynomial bases used by the central and side fits
        cpowers = self._monomial_powers(ndims, degree)
        cexp = np.array(cpowers, dtype=int)
        self._central_power_data = (cpowers, cexp, cexp.sum(axis=1))

        spowers = self._monomial_powers(ndims, min(sub_degree, degree))
        sexp = np.array(spowers, dtype=int)
        self._sub_power_data = (spowers, sexp, sexp.sum(axis=1))

        nsub = nsub or 2**ndims
        self._sub_nterms = len(self._sub_power_data[0])

        # Use an overdetermined directional fit when enough points are present
        self.sub_n = sub_n or max(2*self._sub_nterms, self._sub_nterms + 3)
        self._directions = self._direction_vectors(ndims, nsub)

        # Stencil sizes determine which candidate fits are possible
        self._has_central = self.n >= len(cpowers)
        self._has_dirs = self.n >= self._sub_nterms

        # Select the appropriate nonlinear weighting
        if mode == 'teno':
            self._weights = self._teno_weights
        else:
            self._weights = self._wenoz_weights

    def __call__(self, pts, spts, svals):
        rel = spts - pts[:, None]
        dists = np.linalg.norm(rel, axis=2)

        # Sort each stencil by its distance from the query point
        order = np.argsort(dists, axis=1)
        dists = np.take_along_axis(dists, order, 1)
        rel = np.take_along_axis(rel, order[:, :, None], 1)
        svals = np.take_along_axis(svals, order[:, :, None], 1)

        # Work in a local scaled coordinate system to improve conditioning
        scale = np.maximum(dists[:, -1:], 1.0e-14)
        x = rel / scale[:, :, None]

        cands = []

        # First attempt to build a central fit
        if self._has_central:
            cands.append(self._fit_stencils(x, svals,
                                            self._central_power_data))

        # Then attempt to build directional substencils
        if self._has_dirs:
            ncore = min(max(self._sub_nterms - 1, 2), self.n)
            sub_n = min(self.sub_n, self.n)

            unit = rel / np.maximum(dists, 1e-300)[:, :, None]
            for direction in self._directions:
                align = unit @ direction

                # Penalise misaligned points by up to (1 + 2*dir_bias)
                score = dists*(1 + self.dir_bias*(1 - align))

                # Stencils are sorted so the core is the leading block
                score[:, :ncore] = -np.inf
                idx = np.argpartition(score, sub_n - 1, axis=1)[:, :sub_n]

                xs = np.take_along_axis(x, idx[:, :, None], 1)
                vs = np.take_along_axis(svals, idx[:, :, None], 1)
                cands.append(self._fit_stencils(xs, vs,
                                                self._sub_power_data))

        # Combine the candidate fits with WENO-Z or TENO weights
        if cands:
            vals = np.stack([c[0] for c in cands], axis=1)
            betas = np.stack([c[1] for c in cands], axis=1)
            oks = np.stack([c[2] for c in cands], axis=1)

            weights = self._nonlinear_weights(betas, oks)
            val = (weights[:, None] @ vals)[:, 0]
            nc = oks.any(axis=1)
        else:
            val = np.zeros_like(svals[:, 0])
            nc = np.zeros(len(pts), dtype=bool)

        # Rejected and inadmissible fits revert to the convex IDW fallback
        nc &= self.is_admissible(val.T)

        idw = self._idw(np.zeros_like(pts), rel, svals)
        out = np.where(nc[:, None], val, idw)

        # Exact hits take the value of the coincident source point
        if (ex := dists[:, 0] < self.eps).any():
            out[ex] = svals[ex, 0]

        return out

    @staticmethod
    def _monomial_powers(ndims, degree):
        powers = [p for p in it.product(range(degree + 1), repeat=ndims)
                  if sum(p) <= degree]
        powers.sort(key=lambda p: (sum(p), p))

        return powers

    @staticmethod
    def _monomial_matrix(pts, power_data):
        _, exponents, _ = power_data

        x = np.moveaxis(pts, -1, 0)
        d = int(exponents.max())

        # Powers of each coordinate
        tabs = np.ones((len(x), d + 1, *pts.shape[:-1]))
        for k in range(1, d + 1):
            tabs[:, k] = tabs[:, k - 1]*x

        # Each column is one monomial evaluated at every sample point
        idx = np.arange(len(x))
        cols = [tabs[idx, e].prod(axis=0) for e in exponents]

        return np.stack(cols, axis=-1)

    @staticmethod
    def _wendland_c2(r):
        # Compactly supported C2 kernel: psi(r) = (1-r)_+^4 (4r + 1)
        return np.clip(1 - r, 0.0, None)**4*(4*r + 1)

    @staticmethod
    def _direction_vectors(ndims, nsub):
        # In 2D use evenly spaced angular substencils
        if ndims == 2:
            ang = np.linspace(0.0, 2*np.pi, nsub, endpoint=False)
            return np.column_stack([np.cos(ang), np.sin(ang)])
        # In higher dimensions use axis-balanced corner directions
        else:
            dirs = np.array(list(it.product((-1.0, 1.0), repeat=ndims)))
            dirs /= np.linalg.norm(dirs, axis=1)[:, None]

            return dirs[:nsub]

    def _solve_fits(self, amat, rhs):
        # Scale each column to unit norm to improve the conditioning
        cnorm = np.linalg.norm(amat, axis=1)

        # Skip columns which are negligible, as scaling them amplifies noise
        cmax = cnorm.max(axis=1, keepdims=True)
        cnorm = np.where(cnorm > 1e-12*cmax, cnorm, 1)

        # Apply the scaling, consuming amat as the caller does not reuse it
        amat /= cnorm[:, None, :]

        # Form the normal equations
        gram, grhs = amat.mT @ amat, amat.mT @ rhs

        # Apply a ridge so rank deficient systems can be factored
        gdiag = np.einsum('...ii->...i', gram)
        gdiag += self.ridge

        # Take the Cholesky diagonal as a cheap conditioning estimate
        ldiag = np.einsum('...ii->...i', np.linalg.cholesky(gram))

        # Normal equations square the conditioning, so screen well inside it
        ill = ldiag.min(axis=1) < self.screen

        coeffs = np.linalg.solve(gram, grhs)
        ok = np.ones(len(gram), dtype=bool)

        # Redo the ill-conditioned fits with a singular value decomposition
        if ill.any():
            u, sval, vh = np.linalg.svd(amat[ill], full_matrices=False)
            ok[ill] = sval[:, 0] <= self.cond*sval[:, -1]

            # Truncate modes below the condition limit rather than inverting
            svs = np.where(sval*self.cond > sval[:, :1], sval, np.inf)
            coeffs[ill] = vh.mT @ ((u.mT @ rhs[ill]) / svs[:, :, None])

        # Undo the scaling to recover the monomial basis coefficients
        coeffs /= cnorm[:, :, None]

        return coeffs, ok

    def _fit_stencils(self, x, svals, power_data):
        _, _, orders = power_data

        radii = np.linalg.norm(x, axis=2)
        rmax = np.maximum(radii.max(axis=1, keepdims=True), 1e-14)

        weights = self._wendland_c2(radii / rmax)
        weights = np.maximum(weights, 1e-12)

        # Normalised up front; a uniform row scaling leaves the fit alone
        weights /= weights.sum(axis=1, keepdims=True)

        # The compact weights taper distant points without a hard cutoff
        vdm = self._monomial_matrix(x, power_data)
        sqrtw = np.sqrt(weights)
        amat = sqrtw[:, :, None]*vdm
        rhs = sqrtw[:, :, None]*svals

        coeffs, ok = self._solve_fits(amat, rhs)

        fit = vdm @ coeffs

        # Weighted variance and residual, aggregated across all nvars
        mean = (weights[:, None] @ svals)[:, 0]
        var = (weights*((svals - mean[:, None])**2).sum(axis=2)).sum(axis=1)
        resid = (weights*((fit - svals)**2).sum(axis=2)).sum(axis=1)

        # Higher-order coefficient energy, weighted by total order
        cvar = ((orders[1:, None]*coeffs[:, 1:])**2).sum(axis=(1, 2))

        # Blend residual and higher-order energy into one smoothness beta
        blend = resid + 1e-2*cvar
        norm = var + 1e-14*(mean**2).sum(axis=1)

        nz = norm > 0
        beta = np.where(nz, blend / np.where(nz, norm, 1), 0)

        return coeffs[:, 0], np.where(ok, beta, np.inf), ok

    def _wenoz_weights(self, betas, gamma, tau, eps=1.0e-14):
        alpha = gamma*(1 + (tau[:, None] / (betas + eps))**self.q)
        return alpha / np.maximum(alpha.sum(axis=1, keepdims=True), eps)

    def _teno_weights(self, betas, gamma, tau, eps=1.0e-14):
        alpha = gamma*(1 + (tau[:, None] / (betas + eps))**self.q)
        chi = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), eps)

        # TENO hard-rejects candidates whose normalized weight is too small
        mask = chi > self.ct
        if (none := ~mask.any(axis=1)).any():
            mask[none, betas[none].argmin(axis=1)] = True

        weights = gamma*mask
        return weights / np.maximum(weights.sum(axis=1, keepdims=True), eps)

    def _nonlinear_weights(self, betas, oks):
        ncands = np.maximum(oks.sum(axis=1), 1)
        rcpn = 1 / ncands

        # Bias the linear weights toward an accepted central stencil
        if self._has_central:
            okc = oks[:, 0]
        else:
            okc = np.zeros(len(oks), dtype=bool)

        goth = np.where(okc, (1 - self.gamma0)/np.maximum(ncands - 1, 1), rcpn)
        gamma = np.where(oks, goth[:, None], 0)
        gamma[:, 0] = np.where(okc, self.gamma0, gamma[:, 0])

        # Tau measures how different the candidate smoothness values are
        bmean = np.where(oks, betas, 0).sum(axis=1)*rcpn
        tau = np.where(oks, np.abs(betas - bmean[:, None]), 0).max(axis=1)

        return self._weights(betas, gamma, tau)


class BaseCloudResampler(AlltoallMixin):
    # Interpolation block size, in points
    BLK_SZ = 2**12

    # Peer pre-filter block size, in points
    PREFILTER_BLK_SZ = 2**21

    def __init__(self, pts, solns, interp, progress):
        self.ndims = pts.shape[1]
        self.nvars = solns.shape[1]
        self.interp = interp
        self.progress = progress

        with progress.start('Distribute source points/solutions'):
            self.pts, self.solns, keys = self._distribute_src_points(pts,
                                                                     solns)

        with progress.start('Index source points'):
            # Index the points
            self.pts_tree = self._compute_pts_tree(self.pts)

            # Index the tile bounding boxes of our peer ranks
            self.ebbox_tree = self._compute_bbox_tree(self.pts, keys)

            # Summarise which cells of the domain our peers occupy
            self._prefilter_counts = self._compute_prefilter_counts(self.pts)

        # Track what points we have sent to other ranks
        comm, rank, root = get_comm_rank_root()
        self._fringe_idxs = [set() for i in range(comm.size)]

    def _distribute_src_points(self, pts, solns):
        comm, rank, root = get_comm_rank_root()

        # Obtain the bounding box for our ranks points
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)

        # Reduce over all ranks
        comm.Allreduce(mpi.IN_PLACE, pmin, op=mpi.MIN)
        comm.Allreduce(mpi.IN_PLACE, pmax, op=mpi.MAX)

        # Record the key normalisation for assigning target points
        fact = 2**21 if self.ndims == 3 else 2**32
        self._mort_norm = fact, pmin, (pmax - pmin).max()

        # Normalise our points and compute their Morton codes
        keys = self._morton_keys(pts)

        # Use these codes to sort our points
        sorter = Sorter(comm, keys)

        # Record the segment bounds which map a key to its owning rank
        bounds = np.empty(comm.size, dtype=np.uint64)
        comm.Allgather(sorter.keys[-1:], bounds)
        self._mort_bounds = bounds[:-1]

        # With this redistribute our points and solutions
        return sorter(pts), sorter(solns), sorter.keys

    def _morton_keys(self, pts):
        fact, pmin, ext = self._mort_norm

        # Map the points onto the integer grid the codes are built from
        ipts = np.clip(fact*((pts - pmin) / ext), 0, fact).astype(np.uint64)

        return morton_encode(ipts, [fact]*self.ndims)

    def _compute_pts_tree(self, pts):
        return RTree.from_points(pts, storage='point')

    def _compute_bbox_tree(self, pts, keys):
        comm, rank, root = get_comm_rank_root()

        # Maximum number of tile bounding boxes per rank
        nb = 4096

        # Find the finest aligned Morton block size with at most nb tiles
        for shift in range(0, 64 + self.ndims, self.ndims):
            bkeys = keys >> np.uint64(shift)
            starts = np.flatnonzero(np.diff(bkeys)) + 1
            if len(starts) < nb:
                break

        starts = np.concatenate(([0], starts))

        # Allocate global bounding box array
        bboxes = np.full((comm.size, nb, 2, self.ndims), np.nan)

        # Compute tight bounding boxes for each of our blocks
        if len(pts):
            bboxes[rank, :len(starts), 0] = np.minimum.reduceat(pts, starts)
            bboxes[rank, :len(starts), 1] = np.maximum.reduceat(pts, starts)

        # Exchange
        comm.Allgather(mpi.IN_PLACE, bboxes)

        # Flatten the array, tagging each box with the rank which owns it
        branks = np.repeat(np.arange(comm.size), bboxes.shape[1])
        bmins, bmaxs = np.moveaxis(bboxes, 2, 0).reshape(2, -1, self.ndims)

        # Discard unfilled slots and our own tiles, which are never queried
        sel = ~np.isnan(bmins[:, 0]) & (branks != rank)

        return RTree.from_boxes(bmins[sel], bmaxs[sel], branks[sel])

    def _compute_prefilter_counts(self, pts):
        comm, rank, root = get_comm_rank_root()

        # Coarse grid used to pre-filter neighbour queries
        gbits = 8 if self.ndims == 3 else 12
        self._prefilter_fact = gfact = 2**gbits

        # Flag the cells which contain one of our points
        occ = np.zeros(gfact**self.ndims, dtype=np.int32)
        gpts = self._prefilter_coords(pts)
        occ[np.ravel_multi_index(tuple(gpts.T), (gfact,)*self.ndims)] = 1

        # Count how many ranks hold points in each cell
        own = occ.astype(bool)
        comm.Allreduce(mpi.IN_PLACE, occ, op=mpi.SUM)

        # Cells holding the points of a peer rank
        peer = (occ > own).reshape((gfact,)*self.ndims)

        # Accumulate so any box can be counted in constant time
        counts = np.zeros((gfact + 1,)*self.ndims, dtype=np.int32)
        counts[(slice(1, None),)*self.ndims] = peer
        for i in range(self.ndims):
            counts.cumsum(axis=i, out=counts)

        return counts

    def _prefilter_coords(self, pts):
        gfact, (_, pmin, ext) = self._prefilter_fact, self._mort_norm

        # Normalise the points onto the coarse grid
        gpts = gfact*((pts - pmin) / ext)

        return np.clip(gpts, 0, gfact - 1).astype(np.int32)

    def _prefilter(self, pts, ndists):
        keep, cmins, cmaxs = [], [], []

        # Collect the candidates a block of points at a time
        for bpts, bdst in zip_chunks(self.PREFILTER_BLK_SZ, pts, ndists):
            # Bounding boxes of the neighbours of this block
            bmins, bmaxs = bpts - bdst[:, None], bpts + bdst[:, None]

            # Locate the box corners on the grid, upper face exclusive
            glo = self._prefilter_coords(bmins).T
            ghi = self._prefilter_coords(bmaxs).T + 1

            # Count the peer cells inside each box by inclusion-exclusion
            cnt = np.zeros(len(bpts), dtype=np.int32)
            for i in range(2**self.ndims):
                bits = [(i >> j) & 1 for j in range(self.ndims)]
                idx = [lo if b else hi for b, lo, hi in zip(bits, glo, ghi)]

                cnt += (-1)**sum(bits)*self._prefilter_counts[tuple(idx)]

            # Only boxes reaching the points of a peer need a tree query
            bkeep = cnt > 0

            keep.append(bkeep)
            cmins.append(bmins[bkeep])
            cmaxs.append(bmaxs[bkeep])

        cand = np.flatnonzero(np.concatenate(keep))

        return cand, np.concatenate(cmins), np.concatenate(cmaxs)

    def sample_with_mesh_config(self, mesh, cfg):
        pts, sidxs = [], []

        # Numeric data type
        dtype = np.dtype(cfg.get('backend', 'precision', 'double')).type

        # Interpolate the shape points to the solution points
        for etype in mesh.eidxs:
            op = _ploc_op(etype, len(mesh.spts[etype]), cfg)
            plocs = op @ mesh.spts[etype].reshape(op.shape[1], -1)
            plocs = plocs.reshape(-1, self.ndims)

            pts.append(plocs)
            sidxs.append(len(plocs) + (sidxs[-1] if sidxs else 0))

        # Sample the solution at these solution points
        solns = self.sample_with_pts(np.vstack(pts), dtype)
        solns = np.split(solns, sidxs[:-1])

        # Reshape these solutions into their canonical forms
        esolns = {}
        for (etype, eidxs), soln in zip(mesh.eidxs.items(), solns):
            esoln = soln.reshape(-1, len(eidxs), self.nvars)
            esolns[etype] = esoln.transpose(1, 2, 0)

        return esolns

    def sample_with_pts(self, tpts, dtype):
        comm, rank, root = get_comm_rank_root()

        with self.progress.start('Distribute target points'):
            tpts, tcountdisp, tidx = self._distribute_tgt_pts(tpts)

        with self.progress.start_with_sequence('Sample target points') as ps:
            with ps.start('Find neighbours'):
                didxs, redo, fpdata, fpcnt = self._find_tgt_neighbours(tpts)
                comm.barrier()

            with ps.start('Exchange fringe points'):
                self._exchange_fringe_pts(fpdata, fpcnt)
                comm.barrier()

            with ps.start_with_bar('Interpolate points', counts=False) as pbar:
                tsolns = self._interp_tgt_pts(tpts, didxs, redo, dtype, pbar)
                comm.barrier()

        with self.progress.start('Distribute target samples'):
            return self._distribute_tgt_samples(tsolns, tcountdisp, tidx)

    def _distribute_tgt_pts(self, pts):
        comm, rank, root = get_comm_rank_root()

        # Assign points to the ranks owning their Morton code segments
        keys = self._morton_keys(pts)
        tranks = np.searchsorted(self._mort_bounds, keys)

        # Sort the points according to their rank
        tidx = np.argsort(keys)

        # Distribute the points
        tdisps = np.searchsorted(tranks[tidx], np.arange(comm.size))
        tcount = self._disp_to_count(tdisps, len(tranks))
        rbuf = self._alltoallcv(comm, pts[tidx], tcount, tdisps)

        return *rbuf, np.argsort(tidx)

    def _find_tgt_neighbours(self, pts):
        comm, rank, root = get_comm_rank_root()

        nn = self.interp.n

        # Determine the nearest points to each sample point
        didxs, ndists = self.pts_tree.knn(pts, nn)
        ndists = ndists[:, -1]

        # Narrow the indices as this list lives until the interpolation
        didxs = didxs.astype(np.int32)

        # Exclude the points with no peer sources anywhere near their box
        cand, cmins, cmaxs = self._prefilter(pts, ndists)

        # Determine which peers hold points near each candidate
        iranks, icounts = self.ebbox_tree.intersect(cmins, cmaxs)

        # Pair each intersecting rank with the point which found it
        pidx = np.repeat(cand, icounts.astype(int))

        # A rank may own several intersecting tiles, so deduplicate
        pairs = np.unique(np.stack([iranks, pidx], axis=1), axis=0)

        # Sorting by rank leaves the requests grouped ready to exchange
        fpi = pairs[:, 1]
        fpcnt = np.bincount(pairs[:, 0], minlength=comm.size)

        # Only the points which ask for data can gain new neighbours
        redo = np.unique(fpi)

        # Assemble the fringe point requests
        fpdata = np.hstack([pts[fpi], ndists[fpi, None]])

        return didxs, redo, fpdata, fpcnt

    def _interp_tgt_pts(self, pts, didxs, redo, dtype, pbar):
        comm, rank, root = get_comm_rank_root()

        nn = self.interp.n

        # Fringe insertions can alter any kNN, so query the augmented tree
        if len(redo):
            didxs[redo] = self.pts_tree.knn(pts[redo], nn)[0]

        # Communicator for the ranks on our node
        ncomm = get_node_comm(comm)
        wins = []

        # Stage an array into a node-shared window
        def share(arr):
            win, views = shared_ndarrays(ncomm, arr.shape, arr.dtype)
            views[ncomm.rank][:] = arr
            wins.append(win)

            return views

        pts_v, didxs_v = share(pts), share(didxs)
        spts_v, ssolns_v = share(self.pts), share(self.solns)

        # Allocate the interpolated solution array
        owin, solns_v = shared_ndarrays(ncomm, (len(pts), self.nvars), dtype)
        wins.append(owin)

        # Node-local queue which lets idle ranks steal peer blocks
        queue = SharedWorkQueue(ncomm, len(pts), self.BLK_SZ)

        # Interpolate the points in blocks, tracking global progress
        with queue, mpi_progress(comm, pbar, len(pts)) as tick:
            while (blk := queue.claim()) is not None:
                r, sl = blk
                idxs = didxs_v[r][sl]

                solns_v[r][sl] = self.interp(pts_v[r][sl], spts_v[r][idxs],
                                             ssolns_v[r][idxs])
                tick(sl.stop - sl.start)

        # Copy our own segment out of the shared window
        return solns_v[ncomm.rank].copy()

    def _exchange_fringe_pts(self, sbuf, scount):
        comm, rank, root = get_comm_rank_root()

        nn = self.interp.n

        # Exchange the requests
        rdata, (_, rdisps) = self._alltoallcv(comm, sbuf, scount)

        fidxs, fcount = [], []

        # Split these up on a per-rank basis
        for i, ibboxes in enumerate(np.split(rdata, rdisps[1:])):
            fcnt = len(fidxs)

            # Identify nearby points on our rank
            pts, ndists = ibboxes[:, :self.ndims], ibboxes[:, self.ndims]
            idxs = self.pts_tree.knn(pts, nn, radius=ndists)[0]
            idxs = set(np.unique(idxs[idxs >= 0]).tolist())

            # Exclude points that have already been sent over
            fidxs.extend(idxs - self._fringe_idxs[i])
            self._fringe_idxs[i].update(idxs)

            # Tally up how many points we are sending to this rank
            fcount.append(len(fidxs) - fcnt)

        # Convert these lists to arrays
        fidxs = np.array(fidxs, dtype=int)
        fcount = np.array(fcount, dtype=int)
        fdisps = self._count_to_disp(fcount)

        # Exchange fringe points and solutions
        pts = self._alltoallcv(comm, self.pts[fidxs], fcount, fdisps)[0]
        solns = self._alltoallcv(comm, self.solns[fidxs], fcount, fdisps)[0]

        # Add the received fringe points to our tree
        self.pts_tree.add(np.arange(len(pts)) + len(self.pts), pts)

        # Incorporate this fringe data into our point and solution arrays
        if len(pts):
            self.pts = np.vstack([self.pts, pts])
            self.solns = np.vstack([self.solns, solns])

    def _distribute_tgt_samples(self, tsolns, tcountdisp, tidx):
        comm, rank, root = get_comm_rank_root()

        # Send the samples back to the ranks they came from
        tsolns = self._alltoallcv(comm, tsolns, *tcountdisp)[0]
        return tsolns[tidx]


class NativeCloudResampler(BaseCloudResampler):
    def __init__(self, mesh, soln, interp, progress):
        cpts, csolns = self._concat_pts_solns(mesh, soln)

        super().__init__(cpts, csolns, interp, progress)

    def _concat_pts_solns(self, mesh, soln):
        pts, solns = [], []

        for etype in mesh.eidxs:
            # Interpolate the shape points to the solution points
            op = _ploc_op(etype, len(mesh.spts[etype]), soln.config)
            ploc = op @ mesh.spts[etype].reshape(op.shape[1], -1)
            pts.append(ploc.reshape(-1, mesh.ndims))

            # Extract the solution at the solution points
            supts = soln.data[etype].swapaxes(1, 2)
            solns.append(supts.reshape(-1, supts.shape[2]))

        return np.vstack(pts), np.vstack(solns)
