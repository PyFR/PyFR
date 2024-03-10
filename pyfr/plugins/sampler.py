import numpy as np
from rtree.index import Index, Property

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, init_csv
from pyfr.quadrules import get_quadrule


def _closest_pts(epts, pts):
    ndims = len(pts[0])
    props = Property(dimension=ndims, interleaved=True)
    trees = []

    for e in epts:
        # Build list of solution points for each element type
        ins = [(i, [*p, *p], None) for i, p in enumerate(e.reshape(-1, ndims))]

        # Build tree of solution points for each element type
        trees.append(Index(ins, properties=props))

    for p in pts:
        # Find index of solution point closest to p
        amins = [np.unravel_index(next(t.nearest([*p, *p], 1)), ept.shape[:2])
                 for ept, t in zip(epts, trees)]

        # Find distance of solution point closest to p
        dmins = [np.linalg.norm(e[a] - p) for e, a in zip(epts, amins)]

        # Reduce across element types
        yield min(zip(dmins, range(len(epts)), amins))


def _plocs_to_tlocs(sbasis, spts, plocs, tlocs):
    plocs, itlocs = np.array(plocs), np.array(tlocs)

    # Evaluate the initial guesses
    iplocs = np.einsum('ij,jik->ik', sbasis.nodal_basis_at(itlocs), spts)

    # Iterates
    kplocs, ktlocs = iplocs.copy(), itlocs.copy()

    # Apply three iterations of Newton's method
    for k in range(3):
        jac_ops = sbasis.jac_nodal_basis_at(ktlocs)
        kjplocs = np.einsum('ijk,jkl->kli', jac_ops, spts)
        ktlocs -= np.linalg.solve(kjplocs, kplocs - plocs)

        ops = sbasis.nodal_basis_at(ktlocs)
        np.einsum('ij,jik->ik', ops, spts, out=kplocs)

    # Compute the initial and final distances from the target location
    idists = np.linalg.norm(plocs - iplocs, axis=1)
    kdists = np.linalg.norm(plocs - kplocs, axis=1)

    # Replace any points which failed to converge with their initial guesses
    closer = np.where(idists < kdists)
    ktlocs[closer] = itlocs[closer]
    kplocs[closer] = iplocs[closer]

    return ktlocs, kplocs


class SamplerPlugin(BaseSolnPlugin):
    name = 'sampler'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # List of points to be sampled and format
        self.pts = self.cfg.getliteral(cfgsect, 'samp-pts')
        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # MPI rank responsible for each sample point
        if rank == root:
            ptsrank = []

        # Sample points we're responsible for, grouped by element type
        elepts = [[] for i in range(len(intg.system.ele_map))]

        # Search locations in transformed and physical space
        tlocs, plocs = self._search_pts(intg)

        # For each sample point find our nearest search location
        closest = _closest_pts(plocs, self.pts)

        # Process these points
        for i, (dist, etype, (uidx, eidx)) in enumerate(closest):
            # Reduce over the distance
            _, mrank = comm.allreduce((dist, rank), op=mpi.MINLOC)

            # If we have the closest point then save the relevant info
            if rank == mrank:
                elepts[etype].append((i, eidx, tlocs[etype][uidx]))

            # Note what rank is responsible for the point
            if rank == root:
                ptsrank.append(mrank)

        # Refine
        self._ourpts = ourpts = self._refine_pts(intg, elepts)

        # Send the refined sample locations to the root rank
        ptsplocs = comm.gather([pl for et, ei, pl, op in ourpts], root=root)

        if rank == root:
            nvars = self.nvars

            # Allocate a buffer to store the sampled points
            self._ptsbuf = ptsbuf = np.empty((len(self.pts), self.nvars))

            # Tally up how many points each rank is responsible for
            nptsrank = [len(ploc) for ploc in ptsplocs]

            # Compute the counts and displacements, sans nvars
            ptscounts = np.array(nptsrank, dtype=np.int32)
            ptsdisps = np.cumsum([0] + nptsrank[:-1], dtype=np.int32)

            # Apply the displacements to each ranks points
            miters = [enumerate(ploc, start=pdisp)
                      for ploc, pdisp in zip(ptsplocs, ptsdisps)]

            # With this form the final point (offset, location) list
            self._ptsinfo = [next(miters[pr]) for pr in ptsrank]

            # Form the MPI Gatherv receive buffer tuple
            self._ptsrecv = (ptsbuf, (nvars*ptscounts, nvars*ptsdisps))

            # Open the output file
            self.outf = init_csv(self.cfg, cfgsect, self._header)
        else:
            self._ptsrecv = None

    @property
    def _header(self):
        colnames = ['t', 'x', 'y', 'z'][:self.ndims + 1]

        if self.fmt == 'primitive':
            colnames += self.elementscls.privarmap[self.ndims]
        else:
            colnames += self.elementscls.convarmap[self.ndims]

        return ','.join(colnames)

    def _search_pts(self, intg):
        tlocs, plocs = [], []

        # Use a strictly interior point set
        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }

        for etype, eles in intg.system.ele_map.items():
            pts = get_quadrule(etype, qrule_map[etype], eles.basis.nupts).pts

            tlocs.append(pts)
            plocs.append(eles.ploc_at_np(pts).swapaxes(1, 2))

        return tlocs, plocs

    def _refine_pts(self, intg, elepts):
        elelist = intg.system.ele_map.values()
        ptsinfo = []

        # Loop over all the points for each element type
        for etype, (eles, epts) in enumerate(zip(elelist, elepts)):
            if not epts:
                continue

            idx, eidx, tlocs = zip(*epts)
            spts = eles.eles[:, eidx, :]
            plocs = [self.pts[i] for i in idx]

            # Use Newton's method to find the precise transformed locations
            ntlocs, nplocs = _plocs_to_tlocs(eles.basis.sbasis, spts, plocs,
                                             tlocs)

            # Form the corresponding interpolation operators
            intops = eles.basis.ubasis.nodal_basis_at(ntlocs)

            # Append to the point info list
            ptsinfo.extend(
                (*info, etype) for info in zip(idx, eidx, nplocs, intops)
            )

        # Sort our info array by its original index
        ptsinfo.sort()

        # Strip the index, move etype to the front, and return
        return [(etype, *info) for idx, *info, etype in ptsinfo]

    def _process_samples(self, samps):
        samps = np.array(samps)

        # If necessary then convert to primitive form
        if self.fmt == 'primitive' and samps.size:
            samps = self.elementscls.con_to_pri(samps.T, self.cfg)
            samps = np.array(samps).T

        return np.ascontiguousarray(samps, dtype=float)

    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the solution matrices
        solns = intg.soln

        # Perform the sampling and interpolation
        samples = [op @ solns[et][:, :, ei] for et, ei, _, op in self._ourpts]
        samples = self._process_samples(samples)

        # Gather to the root rank
        comm.Gatherv(samples, self._ptsrecv, root=root)

        # If we're the root rank then output
        if rank == root:
            for off, ploc in self._ptsinfo:
                print(intg.tcurr, *ploc, *self._ptsbuf[off], sep=',',
                      file=self.outf)

            # Flush to disk
            self.outf.flush()
