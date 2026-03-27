import itertools as it
from math import comb

import numpy as np
from rtree.index import Index, Property

from pyfr.mpiutil import AlltoallMixin, Sorter, get_comm_rank_root, mpi
from pyfr.nputil import morton_encode
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


def get_interpolator(name, ndims, opts, order=None):
    return subclass_where(BaseInterpolator, name=name).from_opts(ndims, opts,
                                                                 order=order)


def _ploc_op(etype, nspts, cfg):
    # Shape and associated basis
    shapecls = subclass_where(BaseShape, name=etype)
    basis = shapecls(nspts, cfg)

    # Interpolate the shape points to the solution points
    return basis.sbasis.nodal_basis_at(basis.upts)


def str_group(gpts, nsplit):
    def recursive_tile(pts, depth=0):
        if depth == pts.shape[1]:
            return [pts]

        sidx = np.argsort(pts[:, depth])
        groups = []

        # Recurse in the next dimension
        for chunk in np.array_split(pts[sidx], nsplit):
            groups.extend(recursive_tile(chunk, depth + 1))

        return groups

    return recursive_tile(gpts)


class BaseInterpolator:
    name = None

    int_opts = {}
    float_opts = {}
    enum_opts = {}
    dflt_opts = {}

    @classmethod
    def from_opts(cls, ndims, opts={}, *, order=None):
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

        return cls(ndims, **kwargs)


class IDWInterpolator(BaseInterpolator):
    name = 'idw'

    # Integer options
    int_opts = {'n'}

    # Floating point options
    float_opts = {'rho'}

    # Default options
    dflt_opts = {'n': 0, 'rho': 0}

    def __init__(self, ndims, *, order=None, n=0, rho=0):
        self.n = n or 2**ndims
        self.rho = rho or ndims + 1
        self.eps = 10*np.finfo(float).eps

    def __call__(self, p, spts, svals):
        dists = np.linalg.norm(p - spts, axis=1)
        if dists[ix := dists.argmin()] < self.eps:
            return svals[ix]
        else:
            swts = dists**-self.rho
            swts /= swts.sum()

            return swts @ svals


class WENOInterpolator(BaseInterpolator):
    name = 'weno'

    int_opts = {'n', 'degree', 'sub_degree', 'nsub', 'sub_n'}
    float_opts = {'rho', 'q', 'ct', 'cond', 'dir_bias', 'gamma0'}
    enum_opts = {'mode': {'wenoz': 'wenoz', 'teno': 'teno'}}
    dflt_opts = {'n': 0, 'degree': 0, 'sub_degree': 0, 'nsub': 0,
                 'sub_n': 0, 'rho': 0, 'q': 4, 'ct': 1.0e-3,
                 'cond': 1.0e8, 'dir_bias': 2.5, 'gamma0': 0.85,
                 'mode': 'wenoz'}

    def __init__(self, ndims, *, order=None, n=0, degree=0, sub_degree=0,
                 nsub=0, sub_n=0, rho=0, q=4, ct=1.0e-3, cond=1.0e8,
                 dir_bias=2.5, gamma0=0.85, mode='wenoz'):
        # Auto-derive degree from the source polynomial order
        if not degree:
            degree = min(order, 3) if order else 2

        if not sub_degree:
            sub_degree = max(degree - 1, 1)

        # Auto-derive the stencil size to be ~3x overdetermined
        nterms = comb(ndims + degree, degree)
        if not n:
            n = max(3*nterms, 24 if ndims == 2 else 48)

        self.n = n
        self.q = q
        self.ct = ct
        self.cond = cond
        self.dir_bias = dir_bias
        self.gamma0 = gamma0
        self.eps = 10*np.finfo(float).eps

        # Keep an IDW fallback for exact hits and rejected stencils
        self._idw = IDWInterpolator(ndims, n=self.n, rho=rho or ndims + 1)

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

        # Select the appropriate nonlinear weighting
        if mode == 'teno':
            self._weights = self._teno_weights
        else:
            self._weights = self._wenoz_weights

    def __call__(self, p, spts, svals):
        dists = np.linalg.norm(p - spts, axis=1)
        if dists[ix := dists.argmin()] < self.eps:
            return svals[ix]

        order = np.argsort(dists)
        spts, svals, dists = spts[order], svals[order], dists[order]
        scale = max(dists.max(), 1.0e-14)

        # First attempt to build a central fit
        central = self._fit_central_stencil(p, spts, svals, scale)
        cands = [central] if central is not None else []

        # Then attempt to build directional substencils
        for direction in self._directions:
            idx = self._select_directional_stencil(spts, p, direction)
            if idx is None:
                continue

            fit = self._fit_sub_stencil(p, spts[idx], svals[idx], scale)
            if fit is not None:
                cands.append(fit)

        # If every polynomial stencil is rejected then revert to IDW
        if not cands:
            return self._idw(p, spts, svals)
        # If there is only one candidate fit then return it
        elif len(cands) == 1:
            return cands[0][0]
        # Otherwise combine the candidate fits with WENO-Z or TENO weights
        else:
            betas = np.array([c[1] for c in cands])
            vals = np.vstack([c[0] for c in cands])

            weights = self._nonlinear_weights(betas, central is not None)
            return weights @ vals

    @staticmethod
    def _monomial_powers(ndims, degree):
        powers = [p for p in it.product(range(degree + 1), repeat=ndims)
                  if sum(p) <= degree]
        powers.sort(key=lambda p: (sum(p), p))

        return powers

    @staticmethod
    def _monomial_matrix(pts, power_data):
        _, exponents, _ = power_data

        # Each column is one monomial evaluated at every sample point
        vdm = np.ones((len(pts), len(exponents)))
        for i, exp in enumerate(exponents.T):
            vdm *= pts[:, i:i + 1]**exp

        return vdm

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

    def _select_directional_stencil(self, spts, p, direction):
        vecs = spts - p
        dists = np.linalg.norm(vecs, axis=1)
        if not np.any(nz := dists > 0):
            return None

        unit = np.zeros_like(vecs)
        unit[nz] = vecs[nz] / dists[nz, None]
        align = unit @ direction

        # Penalise misaligned points by up to (1 + 2*dir_bias)
        score = dists*(1 + self.dir_bias*(1 - align))

        # Seed the stencil with the nearest points for robustness
        ncore = min(max(self._sub_nterms - 1, 2), len(dists))
        if ncore < len(dists):
            nearest = np.argpartition(dists, ncore - 1)[:ncore]
            nearest = nearest[np.argsort(dists[nearest])]
        else:
            nearest = np.argsort(dists)

        picks = nearest.tolist()
        seen = set(picks)

        # Then extend it with points aligned with the current direction
        ncand = min(len(score), self.sub_n + len(seen))
        if ncand < len(score):
            cand = np.argpartition(score, ncand - 1)[:ncand]
            cand = cand[np.argsort(score[cand])]
        else:
            cand = np.argsort(score)

        for ix in cand:
            if ix not in seen:
                picks.append(int(ix))
                seen.add(int(ix))

            if len(picks) >= self.sub_n:
                break

        if len(picks) < self._sub_nterms:
            return None
        else:
            return np.array(picks, dtype=int)

    def _fit_stencil_data(self, p, spts, svals, power_data, scale):
        _, _, orders = power_data
        if len(spts) < len(orders):
            return None

        # Work in a local scaled coordinate system to improve conditioning
        x = (spts - p) / scale
        radii = np.linalg.norm(x, axis=1)
        rmax = max(radii.max(), 1e-14)

        weights = self._wendland_c2(radii / rmax)
        weights = np.maximum(weights, 1e-12)

        # The compact weights taper distant points without a hard cutoff
        vdm = self._monomial_matrix(x, power_data)
        sqrtw = np.sqrt(weights)
        amat = sqrtw[:, None]*vdm
        rhs = sqrtw[:, None]*svals

        # A single SVD gives both the condition estimate and the solve
        u, sval, vh = np.linalg.svd(amat, full_matrices=False)

        if sval[0] > self.cond*sval[-1]:
            return None

        coeffs = vh.T @ ((u.T @ rhs) / sval[:, None])
        fit = vdm @ coeffs

        # Weighted variance and residual, aggregated across all nvars
        wsum = weights.sum()
        mean = (weights[:, None]*svals).sum(axis=0) / wsum
        var = (weights*np.sum((svals - mean)**2, axis=1)).sum() / wsum
        resid = (weights*np.sum((fit - svals)**2, axis=1)).sum() / wsum

        # Higher-order coefficient energy, weighted by total order
        coeff_var = np.sum((orders[1:, None]*coeffs[1:])**2)

        # Blend residual and higher-order energy into one smoothness beta
        beta = resid / (var + 1e-14)
        beta += 1e-2*coeff_var / (np.sum(coeffs[0]**2) + 1)

        return coeffs[0], float(beta)

    def _fit_central_stencil(self, p, spts, svals, scale):
        return self._fit_stencil_data(p, spts, svals, self._central_power_data,
                                      scale)

    def _fit_sub_stencil(self, p, spts, svals, scale):
        return self._fit_stencil_data(p, spts, svals, self._sub_power_data,
                                      scale)

    def _linear_weights(self, ncands):
        if ncands == 1:
            return np.array([1.0])

        # Bias the linear weights toward the central stencil
        gamma = np.full(ncands, (1.0 - self.gamma0) / (ncands - 1))
        gamma[0] = self.gamma0

        return gamma

    def _wenoz_weights(self, betas, gamma, eps=1.0e-14):
        # Tau measures how different the candidate smoothness values are
        tau = np.max(np.abs(betas - betas.mean()))
        alpha = gamma*(1 + (tau / (betas + eps))**self.q)
        return alpha / alpha.sum()

    def _teno_weights(self, betas, gamma, eps=1.0e-14):
        tau = np.max(np.abs(betas - betas.mean()))
        alpha = gamma*(1 + (tau / (betas + eps))**self.q)
        chi = alpha / alpha.sum()

        # TENO hard-rejects candidates whose normalized weight is too small
        mask = chi > self.ct
        if not np.any(mask):
            mask[np.argmin(betas)] = True

        weights = gamma*mask
        return weights / weights.sum()

    def _nonlinear_weights(self, betas, has_central):
        # Without a central fit we fall back to uniform linear weights
        if has_central:
            gamma = self._linear_weights(len(betas))
        else:
            gamma = np.full(len(betas), 1 / len(betas))

        return self._weights(betas, gamma)


class BaseCloudResampler(AlltoallMixin):
    def __init__(self, pts, solns, interp, progress):
        self.ndims = pts.shape[1]
        self.nvars = solns.shape[1]
        self.interp = interp
        self.progress = progress

        with progress.start('Distribute source points/solutions'):
            self.pts, self.solns = self._distribute_src_points(pts, solns)

        with progress.start('Index source points'):
            # Index the points
            self.pts_tree = self._compute_pts_tree(self.pts)

            # Index the bounding boxes (inclusives and exclusive of ourself)
            self.ibbox_tree, self.ebbox_tree = self._compute_bbox_trees(
                self.pts
            )

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

        # Normalise our points and compute their Morton codes
        fact = 2**21 if len(pmin) == 3 else 2**32
        ipts = (fact*((pts - pmin) / (pmax - pmin).max())).astype(np.uint64)
        keys = morton_encode(ipts, [fact]*len(pmin))

        # Use these codes to sort our points
        sorter = Sorter(comm, keys)

        # With this redistribute our points and solutions
        return sorter(pts), sorter(solns)

    def _compute_pts_tree(self, pts):
        props = Property(dimension=self.ndims, index_capacity=25,
                         leaf_capacity=25)
        return Index((np.arange(len(pts)), pts, pts), properties=props)

    def _compute_bbox_trees(self, pts):
        comm, rank, root = get_comm_rank_root()

        # Number of splits along each dimension
        ns = 14

        # Allocate global bounding box array
        bboxes = np.full((comm.size, ns**self.ndims, 2, self.ndims), np.nan)

        # Fill out our portion of the array
        for i, spts in enumerate(str_group(pts, ns)):
            if len(spts):
                bboxes[rank, i] = spts.min(axis=0), spts.max(axis=0)

        # Exchange
        comm.Allgather(mpi.IN_PLACE, bboxes)

        # Construct bounding box trees inclusive and exclusive of our rank
        trees = []
        props = Property(dimension=self.ndims, index_capacity=25,
                         leaf_capacity=25)

        for i in (None, comm.rank):
            ins = [(j, (*bmin, *bmax), None)
                   for j, jbboxes in enumerate(bboxes) if j != i
                   for bmin, bmax in jbboxes if not np.isnan(bmin[0])]
            trees.append(Index(ins, properties=props))

        return trees

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
        with self.progress.start('Distribute target points'):
            tpts, tcountdisp, tidx = self._distribute_tgt_pts(tpts)

        with self.progress.start('Sample target points'):
            tsolns = self._sample_tgt_points(tpts, dtype)

        with self.progress.start('Distribute target samples'):
            return self._distribute_tgt_samples(tsolns, tcountdisp, tidx)

    def _distribute_tgt_pts(self, pts):
        comm, rank, root = get_comm_rank_root()

        if comm.size > 1:
            # Attempt to assign points to ranks
            tranks, resid = self._compute_initial_tgt_dist(pts)

            # Exchange residual information between ranks
            rranks = self._compute_resid_ranks(resid)

            # Assign residual points to the rank with the nearest point
            for i, (r, _) in rranks.items():
                tranks[i] = r
        else:
            tranks = np.zeros(len(pts), dtype=int)

        # Sort the points according to their rank
        tidx = np.argsort(tranks)

        # Distribute the points
        tdisps = np.searchsorted(tranks[tidx], np.arange(comm.size))
        tcount = self._disp_to_count(tdisps, len(tranks))
        rbuf = self._alltoallcv(comm, pts[tidx], tcount, tdisps)

        return *rbuf, np.argsort(tidx)

    def _compute_initial_tgt_dist(self, pts):
        comm, rank, root = get_comm_rank_root()

        tranks, off = np.full(len(pts), -1, dtype=int), 0

        # See which ranks, if any, have a claim to each point
        iranks, icounts = self.ibbox_tree.intersection_v(pts, pts)

        # Residual list for target points claimed by multiple ranks
        resid = [[] for i in range(comm.size)]

        for i, (p, c) in enumerate(zip(pts, icounts)):
            # No boxes directly intersect so find the rank with the nearest
            if c == 0:
                tranks[i] = next(self.ibbox_tree.nearest(p, 1))
            # Ideal case of a single-box intersection
            elif c == 1:
                tranks[i] = iranks[off]
            # Multiple intersecting boxes
            else:
                uranks = set(iranks[off:off + c].tolist())

                # All boxes belong to the same rank so assign
                if len(uranks) == 1:
                    tranks[i] = iranks[off]
                # Multiple distinct ranks; defer
                else:
                    for r in uranks:
                        resid[r].append((i, p.tolist()))

            off += c

        return tranks, resid

    def _compute_resid_ranks(self, resid):
        comm, rank, root = get_comm_rank_root()
        rranks, rdists = {}, []

        # Exchange residual points between ranks
        for rresid in comm.alltoall([[p for i, p in r] for r in resid]):
            if rresid:
                # Identify our closest points
                rpts = np.array(rresid)
                _, _, dists = self.pts_tree.nearest_v(
                    rpts, rpts, strict=True, return_max_dists=True
                )

                rdists.append(dists.tolist())
            else:
                rdists.append([])

        # Exchange distance information and reduce
        for r, (re, rd) in enumerate(zip(resid, comm.alltoall(rdists))):
            for (i, _), d in zip(re, rd):
                if (rd := rranks.get(i, None)) is None or d < rd[1]:
                    rranks[i] = (r, d)

        return rranks

    def _sample_tgt_points(self, pts, dtype):
        comm, rank, root = get_comm_rank_root()

        # Allocate the interpolated solution array
        solns = np.empty((len(pts), self.nvars), dtype=dtype)

        nn = self.interp.n
        deferred, off = [], 0
        fpreqs = [[] for i in range(comm.size)]

        # Determine the nearest points to each sample point
        nidxs, _, ndists = self.pts_tree.nearest_v(pts, pts, num_results=nn,
                                                   strict=True,
                                                   return_max_dists=True)

        # Compute the associated bounding boxes
        pmins, pmaxs = pts - ndists[:, None], pts + ndists[:, None]

        # Determine which ranks intersect with these bounding boxes
        iranks, icounts = self.ebbox_tree.intersection_v(pmins, pmaxs)

        for i, (p, count) in enumerate(zip(pts, icounts)):
            # If any other ranks intersect then defer
            if count:
                deferred.append(i)
                for j in np.unique(iranks[off:off + count]):
                    fpreqs[j].append([*p, ndists[i]])

                off += count
            # Otherwise perform the interpolation
            else:
                idxs = nidxs[i*nn:(i + 1)*nn]
                solns[i] = self.interp(p, self.pts[idxs], self.solns[idxs])

        self._exchange_fringe_pts(fpreqs, nn)

        # Process the deferred points
        dpts = pts[deferred]
        didxs, _ = self.pts_tree.nearest_v(dpts, dpts, num_results=nn,
                                           strict=True)
        for i, p, idxs in zip(deferred, dpts, didxs.reshape(-1, nn)):
            solns[i] = self.interp(p, self.pts[idxs], self.solns[idxs])

        comm.barrier()

        # Return the interpolated solution values
        return solns

    def _exchange_fringe_pts(self, frboxes, nn):
        comm, rank, root = get_comm_rank_root()

        # Tally up how many requests we have for each rank
        scount = np.array([len(fr) for fr in frboxes])
        if scount.any():
            sbuf = np.vstack([fr for fr in frboxes if fr])
        else:
            sbuf = np.empty((0, self.ndims + 1))

        # Exchange the requests
        rdata, (_, rdisps) = self._alltoallcv(comm, sbuf, scount)

        fidxs, fcount = [], []

        # Split these up on a per-rank basis
        for i, ibboxes in enumerate(np.split(rdata, rdisps[1:])):
            fcnt = len(fidxs)

            # Identify nearby points on our rank
            pts, ndists = ibboxes[:, :self.ndims], ibboxes[:, self.ndims]
            idxs, _ = self.pts_tree.nearest_v(pts, pts, num_results=nn,
                                              max_dists=ndists, strict=True)
            idxs = set(np.unique(idxs).tolist())

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
        for i, p in enumerate(pts, start=len(self.pts)):
            self.pts_tree.insert(i, p)

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
