# -*- coding: utf-8 -*-

import ast
import os

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin


def _closest_upt(etypes, eupts, p):
    # Compute the distances between each point and p
    dists = [np.linalg.norm(e - p, axis=2) for e in eupts]

    # Get the index of the closest point to p for each element type
    amins = [np.unravel_index(np.argmin(d), d.shape) for d in dists]

    # Dereference to get the actual distances and locations
    dmins = [d[a] for d, a in zip(dists, amins)]
    plocs = [e[a] for e, a in zip(eupts, amins)]

    # Find the minimum across all element types
    return min(zip(dmins, plocs, etypes, amins))


class SamplerPlugin(BasePlugin):
    name = 'sampler'

    def __init__(self, intg, cfgsect):
        from mpi4py import MPI

        super().__init__(intg, cfgsect)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Output frequency
        self.freq = self.cfg.getint(cfgsect, 'freq')

        # List of points to be sampled and format
        self.pts = ast.literal_eval(self.cfg.get(cfgsect, 'samp-pts'))
        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # MPI rank responsible for each point and rank-indexed info
        self._ptsrank = ptsrank = []
        self._ptsinfo = ptsinfo = [[] for i in range(comm.size)]

        # Physical location of the solution points
        plocs = [p.swapaxes(1, 2) for p in intg.system.ele_ploc_upts]

        for p in self.pts:
            # Find the nearest point in our partition
            cp = _closest_upt(intg.system.ele_types, plocs, p)

            # Reduce over all partitions
            mcp, mrank = comm.allreduce(cp, op=MPI.MINLOC)

            # Store the rank responsible along with the info
            ptsrank.append(mrank)
            ptsinfo[mrank].append(mcp[1:])

        # If we're the root rank then open the output file
        if rank == root:
            # Determine the file path
            fname = self.cfg.get(cfgsect, 'file')

            # Append the '.csv' extension
            if not fname.endswith('.csv'):
                fname += '.csv'

            # Open for appending
            self.outf = open(fname, 'a')

            # Output a header if required
            if (os.path.getsize(fname) == 0 and
                self.cfg.getbool(cfgsect, 'header', True)):
                print(self._header, file=self.outf)

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
        if self.fmt == 'primitive':
            samps = self.elementscls.conv_to_pri(samps.T, self.cfg)
            samps = np.array(samps).T

        return samps.tolist()

    def __call__(self, intg):
        # Return if no output is due
        if intg.nsteps % self.freq:
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
