from collections import defaultdict
from functools import partial

from boostree import RTree
import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import autofree, get_comm_rank_root, get_start_end_csize, mpi
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


class PointLocator:
    def __init__(self, mesh, fine_order=6):
        self.mesh = mesh
        self.fine_order = fine_order

        self.dtype = np.dtype([('dist', float), ('cidx', np.int16),
                               ('eidx', np.int64), ('rank', np.int32),
                               ('tloc', float, mesh.ndims)])

    def _reduce_elocs(self, npts, elocator):
        comm, rank, _ = get_comm_rank_root()

        # Allocate the location buffer
        locs = np.zeros(npts, dtype=self.dtype)
        locs['dist'] = np.inf
        locs['rank'] = rank

        # Reduce over each of our element types
        for etype in self.mesh.eidxs:
            cidx = self.mesh.codec.index(f'eles/{etype}')
            pi, di, gi, tl = elocator(etype)
            if not len(pi):
                continue

            # Build new candidate records and compare
            new = locs[pi]
            new['dist'], new['cidx'] = di, cidx
            new['eidx'], new['tloc'] = gi, tl

            if (m := self._minloc_mask(new, locs[pi], ndim=3)).any():
                locs[pi[m]] = new[m]

        # Reduce over all ranks
        self._minloc(comm.Allreduce, mpi.IN_PLACE, locs, ndim=4)

        return locs

    def _locate_nnode(self, pts):
        # Find the nearest node to each query point
        nearest = self._find_closest_node(pts)

        # Callback to identify the nearest elements to these nodes
        def elocator(etype):
            return self._find_closest_element_nnode(etype, pts, nearest)

        return self._reduce_elocs(len(pts), elocator)

    def _locate_bbox(self, pts):
        def elocator(etype):
            return self._find_closest_element_bbox(etype, pts)

        return self._reduce_elocs(len(pts), elocator)

    def locate(self, pts):
        # Attempt to locate points using a bounding box approach
        locs = self._locate_bbox(pts)

        # In extreme cases this approach can leave some valid points
        # un-located; for these we try a nearest node approach
        if np.any(infp := np.isinf(locs['dist'])):
            locs[infp] = self._locate_nnode(pts[infp])

        # Validate
        for i, l in enumerate(locs):
            if l['dist'] == np.inf:
                ploc = ', '.join(str(p) for p in pts[i])
                raise ValueError(f'Unable to locate point ({ploc})')

        return locs

    def locate_disjoint(self, pts):
        comm, rank, _ = get_comm_rank_root()

        # Determine how many points each rank has to locate
        npts = np.empty(comm.size, dtype=int)
        npts[rank] = len(pts)
        comm.Allgather(mpi.IN_PLACE, npts)

        # Iterate through each rank with points
        result = np.empty(0, dtype=self.dtype)
        for r in np.flatnonzero(npts):
            # Broadcast the points from rank r
            if rank == r:
                buf = np.ascontiguousarray(pts, dtype=float)
            else:
                buf = np.empty((npts[r], self.mesh.ndims), dtype=float)
            comm.Bcast(buf, root=r)

            # Perform the location
            locs = self.locate(buf)

            if rank == r:
                result = locs

        return result

    @memoize
    def _get_shape_basis(self, etype, nspts):
        shape = subclass_where(BaseShape, name=etype)
        order = shape.order_from_npts(nspts)

        hpts = shape.std_ele(order)
        lpts = shape.std_ele(1)

        sbasis = get_polybasis(etype, order, hpts)
        lbasis = get_polybasis(etype, 1, lpts)

        lidx = np.argmin(np.linalg.norm(hpts[:, None] - lpts[None], axis=-1),
                         axis=0)

        return shape, sbasis, lbasis, lidx, lpts.mean(axis=0)

    @staticmethod
    def _minloc_mask(src, dst, ndim=None):
        fields = list(src.dtype.fields)[:ndim]

        lmask = src[fields[0]] < dst[fields[0]]
        emask = src[fields[0]] == dst[fields[0]]

        for f in fields[1:]:
            lmask |= emask & (src[f] < dst[f])
            emask &= src[f] == dst[f]

        return lmask

    @memoize
    def _get_minloc_op(self, dtype, ndim):
        def op(pmem, qmem, dt):
            p = np.frombuffer(pmem, dtype=dtype)
            q = np.frombuffer(qmem, dtype=dtype)
            m = self._minloc_mask(p, q, ndim)
            q[m] = p[m]

        return autofree(mpi.Op.Create(op, commute=False))

    def _minloc(self, coll, x, y, ndim=None):
        sbuf = (x, mpi.BYTE) if x is not mpi.IN_PLACE else x
        rbuf = (y, mpi.BYTE)

        coll(sbuf, rbuf, op=self._get_minloc_op(y.dtype, ndim))

    @memoize
    def _get_nodes_off_tree(self):
        comm, _, _ = get_comm_rank_root()

        # Read our portion of the nodes table
        start, end, _ = get_start_end_csize(comm, len(self.mesh.raw['nodes']))
        nodes = self.mesh.raw['nodes'][start:end]['location']

        # Insert these points into a spatial index
        tree = RTree.from_points(nodes)

        return nodes, start, tree

    @memoize
    def _get_bbox_tree(self, etype, *, scale=1.05):
        spts = self.mesh.spts[etype]

        # Compute the bounding boxes
        smin, smax = spts.min(axis=0), spts.max(axis=0)

        # Expand by the scale factor to better account for strong curvature
        expand = (scale - 1)*(smax - smin)
        smin -= expand
        smax += expand

        # Insert these boxes into a spatial index
        return RTree.from_boxes(smin, smax)

    def _find_closest_node(self, pts):
        comm, _, _ = get_comm_rank_root()

        # Query the node index to find our closest node
        nodes, off, tree = self._get_nodes_off_tree()
        nearest = tree.knn(pts, k=1)[0][:, 0]

        buf = np.empty(len(pts), dtype=[('dist', float), ('idx', int)])
        buf['dist'] = np.linalg.norm(pts - nodes[nearest], axis=1)
        buf['idx'] = nearest + off

        self._minloc(comm.Allreduce, mpi.IN_PLACE, buf)

        return buf

    def _find_closest_element_nnode(self, etype, pts, nearest):
        nodes = self.mesh.spts_nodes[etype]

        # See which of our elements contain the nearest node
        eidx, nidx = np.isin(nodes, nearest['idx']).nonzero()

        # Create a map from node number to element indices
        neles = defaultdict(set)
        for ei, ni in zip(eidx, nodes[eidx, nidx]):
            neles[ni].add(ei)

        # Use this to form the set of candidate elements for each point
        pidx, sidx = [], []
        for i, (_, ni) in enumerate(nearest):
            for ei in neles.get(ni, []):
                pidx.append(i)
                sidx.append(ei)

        return self._find_closest_element(etype, pts, np.array(pidx),
                                          np.array(sidx))

    def _find_closest_element_bbox(self, etype, pts):
        # Query the index to find intersecting elements
        tree = self._get_bbox_tree(etype)
        sidx, icounts = tree.intersect(pts, pts)

        pidx = np.repeat(np.arange(len(pts)), icounts.astype(int))
        return self._find_closest_element(etype, pts, pidx, sidx)

    def _find_closest_element(self, etype, pts, pidx, sidx):
        spts = self.mesh.spts[etype]
        curved = self.mesh.spts_curved[etype][sidx]
        eidxs = self.mesh.eidxs[etype]

        # Obtain the closest location inside each of these elements
        dists, tlocs = self._compute_tlocs(etype, spts[:, sidx], pts[pidx],
                                           curved)

        # Group distances by their query point index and reduce
        gidxs = eidxs[sidx]
        order = np.lexsort((gidxs, dists, pidx))
        idx = order[np.unique(pidx[order], return_index=True)[1]]
        return pidx[idx], dists[idx], gidxs[idx], tlocs[idx]

    def _initial_tlocs(self, etype, spts, plocs):
        shape, basis, *_ = self._get_shape_basis(etype, len(spts))
        tpts = shape.std_ele(self.fine_order)
        fop = basis.nodal_basis_at(tpts)

        # Chunk through the query points in batches of 200
        tlocs = np.empty_like(plocs)
        for s in range(0, spts.shape[1], 200):
            e = s + 200

            iplocs = fop @ spts[:, s:e].reshape(len(spts), -1)
            iplocs = iplocs.reshape(len(fop), -1, self.mesh.ndims)

            dists = np.linalg.norm(iplocs - plocs[s:e], axis=2)
            tlocs[s:e] = tpts[dists.argmin(axis=0)]

        return tlocs

    def _compute_tlocs(self, etype, spts, plocs, curved):
        shape, sbasis, lbasis, lidx, cent = self._get_shape_basis(etype,
                                                                  len(spts))

        dists = np.empty(len(plocs))
        tlocs = np.empty_like(plocs)

        # Process linear elements
        if (lin := ~curved).any():
            p, s = plocs[lin], spts[np.ix_(lidx, lin)]
            dists[lin], tlocs[lin] = self._newton_tlocs(
                lbasis, s, p, np.tile(cent, (lin.sum(), 1))
            )

        # Process curved elements
        if curved.any():
            p, s = plocs[curved], spts[:, curved]
            dists[curved], tlocs[curved] = self._newton_tlocs(
                sbasis, s, p, self._initial_tlocs(etype, s, p)
            )

        # Prune invalid points
        dists[~shape.valid_spt(tlocs, tol=1e-4)] = np.inf

        return dists, tlocs

    def _newton_tlocs(self, basis, spts, plocs, ktlocs, niters=20, rtol=1e-10):
        # Helpers for obtaining the nodal basis operators
        nb_op = partial(basis.nodal_basis_at, clean=False)
        jac_nb_op = partial(basis.jac_nodal_basis_at, clean=False)

        # Convergence tolerance criteria
        tol = rtol*np.linalg.norm(np.ptp(spts, axis=0), axis=-1)

        tlocs, dists = ktlocs.copy(), np.empty(len(plocs))
        active = np.arange(len(plocs))
        s, p, k = spts, plocs, ktlocs

        # Evaluate the initial locations in physical space
        kp = np.einsum('ij,jik->ik', nb_op(k), s)

        for _ in range(niters):
            # Take a Newton step
            A = np.einsum('ijk,jkl->kli', jac_nb_op(k), s)
            k -= np.linalg.solve(A, (kp - p)[..., None]).squeeze(axis=-1)

            # Evaluate the physical locations at the new Newton iterate
            kp = np.einsum('ij,jik->ik', nb_op(k), s)

            # Compute the residual and check for convergence
            d = np.linalg.norm(kp - p, axis=1)
            if (done := d < tol).any():
                tlocs[active[done]] = k[done]
                dists[active[done]] = d[done]

                if done.all():
                    return dists, tlocs
                else:
                    keep = ~done
                    active, s = active[keep], s[:, keep]
                    p, k, tol, kp = p[keep], k[keep], tol[keep], kp[keep]

        tlocs[active] = k
        dists[active] = np.linalg.norm(kp - p, axis=1)
        return dists, tlocs


class PointSampler:
    def __init__(self, mesh, spts, slocs=None):
        locf = ['cidx', 'eidx', 'tloc']
        self.mesh = mesh
        self.pts = np.asanyarray(spts, dtype=float)

        if slocs is not None:
            self.locs = slocs[locf]
        else:
            self.locs = PointLocator(mesh).locate(self.pts)[locf]

    def configure_with_intg_nvars(self, intg, nvars):
        # Get the solution bases from the system
        ubases = {etype: eles.basis.ubasis
                  for etype, eles in intg.system.ele_map.items()}

        self._configure_ubases_nvars(ubases, nvars)

    def configure_with_cfg_nvars(self, cfg, nvars):
        ubases = {}

        for etype in self.mesh.eidxs:
            shapecls = subclass_where(BaseShape, name=etype)
            ubases[etype] = shapecls(None, cfg).ubasis

        self._configure_ubases_nvars(ubases, nvars)

    def _configure_ubases_nvars(self, ubases, nvars):
        self.nvars = nvars
        locs = self.locs

        comm, rank, root = get_comm_rank_root()
        ptsrank, pinfo = [], defaultdict(list)

        for j, (etype, eidxs) in enumerate(self.mesh.eidxs.items()):
            eimap = np.argsort(eidxs)
            ubasis = ubases[etype]

            # Filter points which do not belong to this element type
            elocs = locs['cidx'] == self.mesh.codec.index(f'eles/{etype}')
            elocs = elocs.nonzero()[0]

            # See what points we have
            esrch = np.searchsorted(eidxs, locs[elocs]['eidx'], sorter=eimap)
            for i, k, l in zip(elocs, esrch, locs[elocs]):
                if k < eimap.size and eidxs[eimap[k]] == l['eidx']:
                    op = ubasis.nodal_basis_at(l['tloc'][None], clean=False)

                    pinfo[j, eimap[k]].append((len(ptsrank), op))
                    ptsrank.append(i)

        # Group points according to the element they're inside
        self.pinfo, self.pcount = [], len(ptsrank)
        for (et, ei), info in pinfo.items():
            if len(info) == 1:
                self.pinfo.append((et, ei, *info[0]))
            else:
                idxs, ops = zip(*info)
                self.pinfo.append((et, ei, np.array(idxs), np.vstack(ops)))

        # Tell the root rank which points we are responsible for
        ptsrank = comm.gather(ptsrank, root=root)

        if rank == root:
            # Allocate a buffer to store the sampled points
            self._ptsbuf = ptsbuf = np.empty((len(self.pts), nvars))

            # Compute the counts and displacements, sans nvars
            ptscounts = np.array([len(pr) for pr in ptsrank])
            ptsdisps = np.concatenate(([0], np.cumsum(ptscounts[:-1])))

            if ptscounts.sum() != len(self.pts):
                raise RuntimeError('Missing points in solution')

            # Form the MPI Gatherv receive buffer tuple
            self._ptsrecv = (ptsbuf, (nvars*ptscounts, nvars*ptsdisps))

            # Form the reordering list
            self._ptsinv = np.argsort([i for pr in ptsrank for i in pr])

    def gather(self, samples):
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            comm.Gatherv(samples, self._ptsrecv, root=root)
            return self._ptsbuf[self._ptsinv]
        else:
            comm.Gatherv(samples, None, root=root)
            return None

    def etype_pinfo(self):
        etype_flat = defaultdict(list)
        for et, ei, idxs, ops in self.pinfo:
            if np.ndim(idxs) == 0:
                etype_flat[et].append((ei, ops[0], idxs))
            else:
                for idx, op in zip(idxs, ops):
                    etype_flat[et].append((ei, op, idx))

        return {et: tuple(map(np.array, zip(*recs)))
                for et, recs in etype_flat.items()}

    def sample(self, solns, process=None):
        # Perform the sampling
        samples = np.empty((self.pcount, self.nvars))
        for et, ei, idxs, ops in self.pinfo:
            samples[idxs] = ops @ solns[et][:, :, ei]

        # Post-process the samples
        if process:
            samples = np.ascontiguousarray(process(samples.T).T)

        # Gather to the root rank and return
        return self.gather(samples)
