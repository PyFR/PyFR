# -*- coding: utf-8 -*-

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin, init_csv


def _closest_upts_bf(etypes, eupts, pts):
    for p in pts:
        # Compute the distances between each point and p
        dists = [np.linalg.norm(e - p, axis=2) for e in eupts]

        # Get the index of the closest point to p for each element type
        amins = [np.unravel_index(np.argmin(d), d.shape) for d in dists]

        # Dereference to get the actual distances and locations
        dmins = [d[a] for d, a in zip(dists, amins)]
        plocs = [e[a] for e, a in zip(eupts, amins)]

        # Find the minimum across all element types
        yield min(zip(dmins, plocs, etypes, amins))


def _closest_upts_kd(etypes, eupts, pts):
    from scipy.spatial import cKDTree

    # Flatten the physical location arrays
    feupts = [e.reshape(-1, e.shape[-1]) for e in eupts]

    # For each element type construct a KD-tree of the upt locations
    trees = [cKDTree(f) for f in feupts]

    for p in pts:
        # Query the distance/index of the closest upt to p
        dmins, amins = zip(*[t.query(p) for t in trees])

        # Unravel the indices
        amins = [np.unravel_index(i, e.shape[:2])
                 for i, e in zip(amins, eupts)]

        # Dereference to obtain the precise locations
        plocs = [e[a] for e, a in zip(eupts, amins)]

        # Reduce across element types
        yield min(zip(dmins, plocs, etypes, amins))


def _closest_upts(etypes, eupts, pts):
    try:
        # Attempt to use a KD-tree based approach
        yield from _closest_upts_kd(etypes, eupts, pts)
    except ImportError:
        # Otherwise fall back to brute force
        yield from _closest_upts_bf(etypes, eupts, pts)


class SamplerPlugin(BasePlugin):
    name = 'sampler'
    systems = ['*']
    formulations = ['dual', 'std']

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

        # MPI rank responsible for each point
        if rank == root:
            ptsrank = []

        # Sample points our partition is responsible for
        self._ourpts = ourpts = []

        # Physical location of the solution points
        plocs = [p.swapaxes(1, 2) for p in intg.system.ele_ploc_upts]

        # Locate the closest solution points in our partition
        closest = _closest_upts(intg.system.ele_types, plocs, self.pts)

        # Process these points
        for dist, *info in closest:
            # Reduce over the distance
            _, mrank = comm.allreduce((dist, rank), op=get_mpi('minloc'))

            # If we have the closest point then save the relevant info
            if rank == mrank:
                ourpts.append(info)

            # Note what rank is responsible for the point
            if rank == root:
                ptsrank.append(mrank)

        # Collate
        ptsinfo = comm.gather(ourpts, root=root)

        if rank == root:
            nvars = self.nvars

            # Allocate a buffer to store the sampled points
            self._ptsbuf = ptsbuf = np.empty((len(self.pts), self.nvars))

            # Tally up how many points each rank is responsible for
            nptsrank = [len(pi) for pi in ptsinfo]

            # Compute the counts and displacements, sans nvars
            ptscounts = np.array(nptsrank, dtype=np.int32)
            ptsdisps = np.cumsum([0] + nptsrank[:-1], dtype=np.int32)

            # Apply the displacements to each ranks points
            miters = [enumerate(pinfo, start=pdisp)
                      for pinfo, pdisp in zip(ptsinfo, ptsdisps)]

            # With this form the final point info list
            self._ptsinfo = [(intg.rallocs.mprankmap[pr], *next(miters[pr]))
                             for pr in ptsrank]

            # Form the MPI Gatherv receive buffer tuple
            self._ptsrecv = (ptsbuf, (nvars*ptscounts, nvars*ptsdisps))

            # Open the output file
            self.outf = init_csv(self.cfg, cfgsect, self._header)
        else:
            self._ptsrecv = None

    @property
    def _header(self):
        colnames = ['t'] + ['x', 'y', 'z'][:self.ndims]
        colnames += ['prank', 'etype', 'uidx', 'eidx']

        if self.fmt == 'primitive':
            colnames += self.elementscls.privarmap[self.ndims]
        else:
            colnames += self.elementscls.convarmap[self.ndims]

        return ','.join(colnames)

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

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))

        # Sample the solution matrices at these points
        samples = [solns[et][ui, :, ei] for _, et, (ui, ei) in self._ourpts]
        samples = self._process_samples(samples)

        # Gather to the root rank
        comm.Gatherv(samples, self._ptsrecv, root=root)

        # If we are the root rank then output
        if rank == root:
            for prank, off, (ploc, etype, idx) in self._ptsinfo:
                print(intg.tcurr, *ploc, prank, etype, *idx,
                      *self._ptsbuf[off], sep=',', file=self.outf)

            # Flush to disk
            self.outf.flush()
