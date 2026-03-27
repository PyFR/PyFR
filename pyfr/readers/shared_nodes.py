from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from pyfr.mpiutil import AlltoallMixin, get_comm_rank_root
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


@dataclass
class SharedNodes:
    by_rank: dict = field(default_factory=dict)
    by_node: dict = field(default_factory=dict)


class SharedNodesFinder(AlltoallMixin):
    def __init__(self, eles, node_idxs, node_valency):
        self.eles = eles
        self.node_idxs = node_idxs
        self.node_valency = node_valency

        self.comm, self.rank, _ = get_comm_rank_root()

    def compute(self):
        if self.comm.size == 1:
            return SharedNodes()

        # Find boundary nodes (shared with other ranks)
        bnodes = self._find_boundary_nodes()

        # Rendezvous: send to home ranks, aggregate, scatter back
        rresp = self._rendezvous_exchange(bnodes)

        # Parse responses into SharedNodes
        return self._parse_responses(rresp)

    def _get_corner_indices(self, etype, nspts):
        # Find corner node indices by matching against linear element coords
        shapecls = subclass_where(BaseShape, name=etype)
        order = shapecls.order_from_npts(nspts)

        spts = np.array(shapecls.std_ele(order))
        linspts = np.array(shapecls.std_ele(1))

        return np.argmin(np.linalg.norm(spts - linspts[:, None], axis=2),
                         axis=1)

    def _find_boundary_nodes(self):
        # Collect corner nodes from all elements
        cnodes = []
        for etype, einfo in self.eles.items():
            nspts = einfo['nodes'].shape[1]
            cidxs = self._get_corner_indices(etype, nspts)
            cnodes.append(einfo['nodes'][:, cidxs].ravel())

        cnodes = np.concatenate(cnodes)
        lnodes, lcounts = np.unique(cnodes, return_counts=True)

        # Boundary nodes are those with a local count less than valency
        mval = self.node_valency[np.searchsorted(self.node_idxs, lnodes)]
        return lnodes[lcounts < mval]

    def _rendezvous_exchange(self, bnodes):
        comm = self.comm

        # Send nodes to their home ranks
        rnodes, rranks = self._send_to_home_ranks(bnodes)

        # Aggregate and build response buffers
        if rnodes.size == 0:
            rdata = np.empty(0, dtype=int)
            rcounts = np.zeros(comm.size, dtype=np.int32)
        else:
            rdata, rcounts = self._build_responses(rnodes, rranks)

        return self._alltoallcv(comm, rdata, rcounts)[0]

    def _send_to_home_ranks(self, bnodes):
        comm = self.comm

        # Hash each node to assign it to a home rank
        hranks = (2654435761*bnodes % comm.size).astype(np.int32)

        # Sort by home rank
        sidx = np.argsort(hranks)
        snodes = bnodes[sidx]
        shomes = hranks[sidx]

        # Compute send counts/displacements and exchange
        scounts = np.bincount(shomes, minlength=comm.size).astype(np.int32)
        rnodes, (rcounts, _) = self._alltoallcv(comm, snodes, scounts)

        # Reconstruct source ranks from counts
        rranks = np.repeat(np.arange(comm.size, dtype=np.int32), rcounts)

        return rnodes, rranks

    def _build_responses(self, rnodes, rranks):
        order = np.argsort(rnodes)
        rnodes, rranks = rnodes[order], rranks[order]

        # Find group boundaries where node changes
        unodes, gstarts = np.unique(rnodes, return_index=True)
        gsizes = np.diff(gstarts, append=rnodes.size)

        # Pack messages as [node, n_sharers, rank_0, rank_1, ...]
        plens = 2 + gsizes
        packed = np.empty(plens.sum(), dtype=int)

        pstarts = self._count_to_disp(plens)
        packed[pstarts] = unodes
        packed[pstarts + 1] = gsizes

        # Scatter ranks into packed array
        offsets = np.repeat(2 + 2*np.arange(len(gsizes)), gsizes)
        packed[offsets + np.arange(rranks.size)] = rranks

        # Duplicate and reorder messages by destination rank for Alltoallv
        s2m = np.repeat(np.arange(len(gsizes)), gsizes)
        order = np.argsort(rranks)
        sdest, smsg = rranks[order], s2m[order]

        mlens, sstarts = plens[smsg], pstarts[smsg]
        offsets = np.repeat(sstarts - self._count_to_disp(mlens), mlens)
        rdata = packed[offsets + np.arange(len(offsets))]

        # Count elements per destination
        rcounts = np.zeros(self.comm.size, dtype=np.int32)
        np.add.at(rcounts, sdest, mlens)

        return rdata, rcounts

    def _parse_responses(self, rresp):
        by_rank = defaultdict(list)
        by_node = defaultdict(list)

        pos = 0
        while pos < rresp.size:
            node, n = int(rresp[pos]), int(rresp[pos + 1])
            ranks = rresp[pos + 2:pos + 2 + n]

            for r in ranks[ranks != self.rank].tolist():
                by_rank[r].append(node)
                by_node[node].append(r)

            pos += 2 + n

        by_rank = {r: np.array(v) for r, v in by_rank.items()}
        by_node = {n: np.array(r) for n, r in by_node.items()}

        return SharedNodes(by_rank, by_node)
