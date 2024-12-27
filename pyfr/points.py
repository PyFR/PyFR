from collections import defaultdict

import numpy as np
from rtree.index import Index, Property

from pyfr.cache import memoize
from pyfr.mpiutil import autofree, get_comm_rank_root, get_start_end_csize, mpi
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


class PointLocator:
    def __init__(self, mesh, fine_order=6):
        self.mesh = mesh
        self.fine_order = fine_order

    def locate(self, pts):
        comm, rank, root = get_comm_rank_root()

        # Allocate the location buffer
        dtype = [('dist', float), ('cidx', np.int16), ('eidx', np.int64),
                 ('tloc', float, self.mesh.ndims)]
        locs = np.zeros(len(pts), dtype=dtype)
        locs['dist'] = np.inf

        # Find the nearest node to each query point
        nearest = self._find_closest_node(pts)

        # Reduce over each of our element types
        for etype, eidxs in self.mesh.eidxs.items():
            cidx = self.mesh.codec.index(f'eles/{etype}')

            eclosest = self._find_closest_element(etype, pts, nearest)
            for i, (dist, eidx, tloc) in eclosest.items():
                l = locs[i]

                if dist < l['dist']:
                    l['dist'], l['tloc'] = dist, tloc
                    l['cidx'], l['eidx'] = cidx, eidxs[eidx]

        # Reduce
        self._minloc(comm.Allreduce, mpi.IN_PLACE, locs, ndim=3)

        # Validate
        if rank == root:
            for i, l in enumerate(locs):
                if l['dist'] == np.inf:
                    ploc = ', '.join(str(p) for p in pts[i])
                    raise ValueError(f'Unable to locate point ({ploc})')

        return locs

    @memoize
    def _get_shape_basis(self, etype, nspts):
        shape = subclass_where(BaseShape, name=etype)
        order = shape.order_from_npts(nspts)
        basis = get_polybasis(etype, order + 1, shape.std_ele(order))

        return shape, basis

    def _minloc(self, coll, x, y, ndim=None):
        dtype = y.dtype
        fields = list(dtype.fields)[:ndim]

        def op(pmem, qmem, dt):
            p = np.frombuffer(pmem, dtype=dtype)
            q = np.frombuffer(qmem, dtype=dtype)

            mask = p[fields[0]] < q[fields[0]]
            for i, f in enumerate(fields[1:], start=1):
                fmask = p[f] < q[f]
                for g in fields[:i]:
                    fmask &= p[g] == q[g]

                mask |= fmask

            q[mask] = p[mask]

        sbuf = (x, mpi.BYTE) if x is not mpi.IN_PLACE else x
        rbuf = (y, mpi.BYTE)

        coll(sbuf, rbuf, op=autofree(mpi.Op.Create(op, commute=False)))

    def _find_closest_node(self, pts):
        comm, rank, root = get_comm_rank_root()

        # Read our portion of the nodes table
        start, end, _ = get_start_end_csize(comm, len(self.mesh.raw['nodes']))
        nodes = self.mesh.raw['nodes'][start:end]['location']

        # Insert these points into a spatial index
        props = Property(dimension=self.mesh.ndims, interleaved=True)
        ins = ((i, [*p, *p], None) for i, p in enumerate(nodes))
        idx = Index(ins, properties=props)

        # Query the index to find our closest node
        nearest = np.array([next(idx.nearest([*p, *p], 1)) for p in pts])

        buf = np.empty(len(pts), dtype=[('dist', float), ('idx', int)])
        buf['dist'] = np.linalg.norm(pts - nodes[nearest], axis=1)
        buf['idx'] = nearest + start

        self._minloc(comm.Allreduce, mpi.IN_PLACE, buf)

        return buf

    def _find_closest_element(self, etype, pts, nearest):
        spts = self.mesh.spts[etype]
        nodes = self.mesh.spts_nodes[etype]

        # See which of our elements contain the nearest node
        eidx, nidx = np.isin(nodes, nearest['idx']).nonzero()

        # Create a map from node number to element indices
        neles = defaultdict(set)
        for ei, ni in zip(eidx, nodes[eidx, nidx]):
            neles[ni].add(ei)

        # Use this to form the set of candidate elements for each point
        pidx, sidx = [], []
        for i, (di, ni) in enumerate(nearest):
            for ei in neles.get(ni, []):
                pidx.append(i)
                sidx.append(ei)

        # Obtain the closest location inside each of these elements
        dists, tlocs = self._compute_tlocs(etype, spts[:, sidx], pts[pidx])

        # For each query point identify the most promising element
        closest = {}
        for i, (pi, dist, tloc) in enumerate(zip(pidx, dists, tlocs)):
            if pi not in closest or dist < closest[pi][0]:
                closest[pi] = (dist, i)

        pidx = list(closest)
        tidx = [i for d, i in closest.values()]
        sidx = [sidx[i] for i in tidx]

        return dict(zip(pidx, zip(dists[tidx], sidx, tlocs[tidx])))

    def _initial_tlocs(self, etype, spts, plocs):
        shape, basis = self._get_shape_basis(etype, len(spts))

        # Obtain a fine sampling of points inside each element
        fop = basis.nodal_basis_at(shape.std_ele(self.fine_order))
        fpts = fop @ spts.reshape(len(spts), -1)
        fpts = fpts.reshape(len(fop), *spts.shape[1:])

        # Find the closest fine sample point to each query point
        dists = np.linalg.norm(fpts - plocs, axis=2)
        amins = np.unravel_index(dists.argmin(axis=0), fpts.shape[:2])

        return fpts[amins]

    def _compute_tlocs(self, etype, spts, plocs):
        shape, basis = self._get_shape_basis(etype, len(spts))

        # Evaluate the initial guesses
        ktlocs = self._initial_tlocs(etype, spts, plocs)
        kplocs = np.einsum('ij,jik->ik',
                           basis.nodal_basis_at(ktlocs, clean=False), spts)

        # Apply three iterations of Newton's method
        for k in range(3):
            jac_ops = basis.jac_nodal_basis_at(ktlocs, clean=False)

            A = np.einsum('ijk,jkl->kli', jac_ops, spts)
            b = kplocs - plocs
            ktlocs -= np.linalg.solve(A, b[..., None]).squeeze()

            ops = basis.nodal_basis_at(ktlocs, clean=False)
            np.einsum('ij,jik->ik', ops, spts, out=kplocs)

        # Compute the final distances
        dists = np.linalg.norm(kplocs - plocs, axis=1)

        # Prune invalid points
        for i, t in enumerate(ktlocs):
            if not shape.valid_spt(t):
                dists[i] = np.inf

        return dists, ktlocs


class PointSampler:
    def __init__(self, mesh, spts, slocs=None):
        locf = ['cidx', 'eidx', 'tloc']
        self.mesh = mesh

        # Named point set
        if isinstance(spts, str):
            comm, rank, root = get_comm_rank_root()

            if rank == root:
                sinfo = mesh.raw[f'plugins/sampler/{spts}'][:]
            else:
                sinfo = None

            sinfo = comm.bcast(sinfo, root=root)

            self.pts, self.locs = sinfo['ploc'], sinfo[locf]
        # Points with location data
        elif slocs is not None:
            self.pts, self.locs = spts, slocs[locf]
        # Points without location data
        else:
            self.pts = np.array(spts)
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

    def sample(self, solns, process=None):
        comm, rank, root = get_comm_rank_root()

        # Perform the sampling
        samples = np.empty((self.pcount, self.nvars))
        for et, ei, idxs, ops in self.pinfo:
            samples[idxs] = ops @ solns[et][:, :, ei]

        # Post-process the samples
        if process:
            samples = np.ascontiguousarray(process(samples))

        # Gather to the root rank and return
        if rank == root:
            comm.Gatherv(samples, self._ptsrecv, root=root)
            return self._ptsbuf[self._ptsinv]
        else:
            comm.Gatherv(samples, None, root=root)
            return None
