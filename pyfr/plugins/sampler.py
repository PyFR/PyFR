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

        # MPI rank responsible for each point and rank-indexed info
        self._ptsrank = ptsrank = []
        self._ptsinfo = ptsinfo = [[] for i in range(comm.size)]

        # Physical location of the solution points
        plocs = [p.swapaxes(1, 2) for p in intg.system.ele_ploc_upts]

        # Locate the closest solution points in our partition
        closest = _closest_upts(intg.system.ele_types, plocs, self.pts)

        # Process these points
        for cp in closest:
            # Reduce over the distance
            _, mrank = comm.allreduce((cp[0], rank), op=get_mpi('minloc'))

            # Store the rank responsible along with its info
            ptsrank.append(mrank)
            ptsinfo[mrank].append(
                comm.bcast(cp[1:] if rank == mrank else None, root=mrank)
            )

        # If we're the root rank then open the output file
        if rank == root:
            self.outf = init_csv(self.cfg, cfgsect, self._header)

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

        return samps.tolist()

    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))

        # Points we're responsible for sampling
        ourpts = self._ptsinfo[comm.rank]

        # Sample the solution matrices at these points
        samples = [solns[et][ui, :, ei] for _, et, (ui, ei) in ourpts]
        samples = self._process_samples(samples)

        # Gather to the root rank to give a list of points per rank
        samples = comm.gather(samples, root=root)

        # If we're the root rank then output
        if rank == root:
            # Collate
            iters = [zip(pi, sp) for pi, sp in zip(self._ptsinfo, samples)]

            for mrank in self._ptsrank:
                # Unpack
                (ploc, etype, idx), samp = next(iters[mrank])

                # Determine the physical mesh rank
                prank = intg.rallocs.mprankmap[mrank]

                # Prepare the output row
                row = [[intg.tcurr], ploc, [prank, etype], idx, samp]
                row = ','.join(str(r) for rp in row for r in rp)

                # Write
                print(row, file=self.outf)

            # Flush to disk
            self.outf.flush()
