import math

import numpy as np

from pyfr.partitioners.baseline.csr import CSRGraph
from pyfr.partitioners.baseline.fm import (GreedyKWayRefiner, TwoWayFMRefiner,
                                           boundary_gains, can_move)
from pyfr.partitioners.baseline.islands import IslandRepair


class GraphPartitioner:
    @staticmethod
    def _targets(partwts, graph):
        return partwts[:, None]*graph.vwts.sum(axis=0)

    def _bounds(self, partwts, graph, imbalance=None):
        imbalance = imbalance or self.imbalance
        targets = self._targets(partwts, graph)
        upper = np.maximum(targets*(1 + imbalance), np.ceil(targets))
        lower = max(1 - imbalance, 0)*targets
        return targets, upper, lower

    def __init__(self, *, seed=2079, nrefine=10, nstarts=4, imbalance=0.01,
                 coarsen_min_mult=20, coarsen_min_abs=500, coarsen_stop=0.8):
        self.nrefine, self.nstarts = nrefine, nstarts
        self.imbalance = imbalance
        self.coarsen_min_mult = coarsen_min_mult
        self.coarsen_min_abs = coarsen_min_abs
        self.coarsen_stop = coarsen_stop
        self.rng = np.random.default_rng(seed)

        # Only rebalance the finest level beyond twice the permitted imbalance
        self._fine_btol = imbalance

    def partition(self, graph, partwts, ptype='kway'):
        nv, ne = len(graph.vtab) - 1, len(graph.etab)
        idt = np.int32 if max(nv, ne) < 2**31 else np.int64

        vtab = graph.vtab.astype(idt)
        etab = graph.etab.astype(idt)
        ewts = graph.ewts.astype(np.int32)
        vwts = np.ascontiguousarray(graph.vwts, dtype=np.int32)

        partwts = np.asarray(partwts, dtype=float)
        partwts /= partwts.sum()

        nparts = len(partwts)
        if nparts == 1:
            return np.zeros(nv, dtype=np.int32)

        graph = CSRGraph(vtab, etab, ewts, vwts)

        # Partition by recursive bisection or the multilevel k-way method
        if ptype == 'rb':
            # Share the permitted imbalance between the bisection levels
            imbalance = self.imbalance / math.log2(nparts)
            return self._rb_kway(graph, partwts, imbalance)
        else:
            return self._multilevel_kway(graph, partwts)

    def _multilevel_kway(self, graph, partwts):
        nv, nparts = graph.nv, len(partwts)

        # Build coarsening hierarchy
        min_size = max(nv // (40*max(math.log2(nparts), 1)),
                       self.coarsen_min_mult*nparts, self.coarsen_min_abs)
        max_vwt = 1.5*graph.vwts.sum(axis=0) / min_size
        graphs, matches = self.hierarchy(graph, min_size, max_vwt)

        # Initial k-way partition at coarsest level
        parts = self._rb_kway(graphs[-1], partwts)

        # Refine the initial partition
        nrefine = self.nrefine
        c_btol = 0.02 if matches else self._fine_btol
        self._refine_level(parts, graphs[-1], partwts, nrefine, c_btol)

        # Uncoarsen with k-way refinement
        return self._descend(parts, graphs, matches, partwts, nrefine)

    def hierarchy(self, graph, min_size, max_vwt=None):
        graphs, matches = [graph], []
        cstop = self.coarsen_stop

        while True:
            fine = graphs[-1]
            if fine.nv <= min_size:
                break

            result = self._coarsen(fine, max_vwt)
            if result is None or result[0].nv >= cstop*fine.nv:
                break

            graphs.append(result[0])
            matches.append(result[1])

        return graphs, matches

    def score(self, parts, graph, partwts):
        # Excess over the bounds, then number of pieces, then the cut
        cut = graph.cut(parts)
        ncomp = graph.components(parts).max() + 1
        if graph.ncon > 1:
            targets, upper, _ = self._bounds(partwts, graph)
            pw = graph.part_wts(parts, len(partwts))
            excess = (pw - upper) / np.maximum(targets, 1e-10)
            return (max(0, excess.max()), ncomp, cut)
        else:
            return (0, ncomp, cut)

    def _descend(self, parts, graphs, matches, partwts, nrefine):
        for i in range(len(matches) - 1, -1, -1):
            parts = parts[matches[i]]
            self._refine_level(parts, graphs[i], partwts, nrefine,
                               btol=0.02 if i else self._fine_btol)

        return parts

    def _refine_level(self, parts, graph, partwts, nrefine, btol=0.02):
        if graph.nv <= len(partwts) or graph.ne == 0:
            return

        self._greedy_refine(parts, graph, partwts)

        if nrefine > 0:
            self.refine_kway(parts, graph, partwts, nrefine=nrefine,
                             btol=btol)

    def eliminate_components(self, parts, graph, partwts, maximb=1.2):
        nv, nparts = graph.nv, len(partwts)
        if nv <= nparts or graph.ne == 0:
            return 0

        targets, upper, _ = self._bounds(partwts, graph)

        # Repeat while annexed chains keep creating new islands
        nmoved = 0
        for _ in range(3):
            comp = graph.components(parts)
            islands = IslandRepair(graph, parts, comp, targets, upper, maximb)
            if islands.keep.all():
                break

            nmoved += islands.run()
            if not islands.nannexed:
                break

        return nmoved

    def _rb_kway(self, graph, partwts, imbalance=None):
        nv, nparts = graph.nv, len(partwts)

        if nv <= nparts:
            return np.arange(nv, dtype=np.int32) % nparts

        # Split partition weights into two balanced groups
        half = partwts.sum() / 2
        cumwt = np.cumsum(partwts)
        split = int(np.searchsorted(cumwt, half, side='right'))
        split = max(1, min(split, nparts - 1))
        left_wt = partwts[:split].sum()
        ratio = left_wt / partwts.sum()

        # Bisect the graph
        bparts = self._ml_bisect(graph, ratio, imbalance)

        # Map partition 0 -> left group, partition 1 -> right group
        parts = np.empty(nv, dtype=np.int32)

        for side, prange in [(0, range(split)), (1, range(split, nparts))]:
            vidx = np.flatnonzero(bparts == side)
            npk = len(prange)

            if npk == 1:
                parts[vidx] = prange.start
            else:
                sub = graph.subgraph(vidx)
                sub_pw = partwts[prange.start:prange.stop]
                sub_parts = self._rb_kway(sub, sub_pw, imbalance)

                # Remap local partition IDs to global
                parts[vidx] = sub_parts + prange.start

        return parts

    def _coarsen(self, graph, max_vwt=None):
        if graph.ne == 0:
            return None

        # Forbid matches which would make a vertex too heavy to balance
        if max_vwt is not None:
            vwts = graph.vwts
            pair_wts = vwts[graph.esrc] + vwts[graph.etab]
            allowed = ~(pair_wts > max_vwt).any(axis=1)
        else:
            allowed = None

        # Heavy edge matching
        match = self._heavy_edge_match(graph, allowed)

        return graph.contract(match), match

    def _heavy_edge_match(self, graph, allowed=None):
        vtab, etab, ewts = graph.vtab, graph.etab, graph.ewts
        nv, ne, edge_src = graph.nv, graph.ne, graph.esrc
        idt = etab.dtype

        partner = np.full(nv, -1, dtype=idt)
        arange_nv = np.arange(nv)

        # KaHIP-style inner/outer edge rating: ewt / (deg_u * deg_v)
        # favours edges between low-degree vertices, encouraging
        # merging of tightly-coupled clusters
        degrees = np.diff(vtab).astype(float)
        base_scores = ewts / (degrees[edge_src]*degrees[etab])
        noise_scale = 0.05*base_scores.max()

        # Preallocate per-round arrays to avoid repeated allocation
        scores = np.empty(ne, dtype=float)
        best_score = np.empty(nv, dtype=float)
        proposal = np.empty(nv, dtype=idt)
        has_edges = np.diff(vtab) > 0
        starts = vtab[:-1][has_edges]

        for _ in range(8):
            unmatched = partner < 0
            um_mask = unmatched[edge_src] & unmatched[etab]
            if allowed is not None:
                um_mask &= allowed
            if not um_mask.any():
                break

            # Score edges with random tiebreaker
            scores[:] = base_scores
            scores += self.rng.random(ne)*noise_scale
            scores[~um_mask] = -np.inf

            # Each unmatched vertex proposes to best unmatched neighbor;
            # reduceat exploits CSR order (edge_src is sorted)
            best_score[:] = -np.inf
            best_score[has_edges] = np.maximum.reduceat(scores, starts)

            is_best = um_mask & (scores >= best_score[edge_src] - 1e-10)
            best_eidx = np.flatnonzero(is_best)
            _, first = np.unique(edge_src[best_eidx], return_index=True)
            selected = best_eidx[first]

            proposal[:] = -1
            proposal[edge_src[selected]] = etab[selected]

            # Accept mutual proposals
            prop_safe = np.where(proposal >= 0, proposal, 0)
            mutual = (proposal >= 0) & (proposal[prop_safe] == arange_nv)
            mutual_v = np.flatnonzero(mutual)
            partner[mutual_v] = proposal[mutual_v]

            # Accept one-sided proposals where the target is unmatched
            self._accept_one_sided(partner, proposal, mutual)

        # Build coarse vertex ID mapping from matched pairs
        return self._coarse_ids(partner)

    @staticmethod
    def _accept_one_sided(partner, proposal, mutual):
        proposers = np.flatnonzero((proposal >= 0) & ~mutual & (partner < 0))
        targets = proposal[proposers]
        keep = partner[targets] < 0
        proposers, targets = proposers[keep], targets[keep]

        _, idx = np.unique(targets, return_index=True)
        proposers, targets = proposers[idx], targets[idx]

        # A vertex that is both a proposer and a target
        # cannot serve both roles in vectorized assignment;
        # drop the entry where it appears as a target
        keep = ~np.isin(targets, proposers)
        proposers, targets = proposers[keep], targets[keep]

        partner[proposers] = targets
        partner[targets] = proposers

    @staticmethod
    def _coarse_ids(partner):
        nv, idt = len(partner), partner.dtype

        is_owner = (partner >= 0) & (np.arange(nv) < partner)
        owners = np.flatnonzero(is_owner)
        n_pairs = len(owners)

        match = np.full(nv, -1, dtype=idt)
        pair_ids = np.arange(n_pairs, dtype=idt)
        match[owners] = pair_ids
        match[partner[owners]] = pair_ids

        # Unmatched vertices become singleton coarse vertices
        unmatched = match < 0
        match[unmatched] = np.arange(n_pairs, n_pairs + unmatched.sum(),
                                     dtype=idt)

        return match

    def _ml_bisect(self, graph, target_ratio, imbalance=None):
        nv = graph.nv
        partwts = np.array([target_ratio, 1 - target_ratio])

        # Coarsen down to a handful of vertices which can still be balanced
        min_size = 2*self.coarsen_min_mult
        if graph.ncon > 1:
            max_vwt = 0.05*min(partwts)*graph.vwts.sum(axis=0)
        else:
            max_vwt = None
        graphs, matches = self.hierarchy(graph, min_size, max_vwt)

        # Candidate bisections of the coarsest graph
        cands = self._bisect(graphs[-1], target_ratio, imbalance)

        # Uncoarsen with two-way refinement, pruning to one candidate
        for i in range(len(matches) - 1, -1, -1):
            level, prev = graphs[i], graphs[i + 1]

            # Keep only the best candidate once the levels become expensive
            if len(cands) > 1 and 8*level.nv > nv:
                cands = [self.best_parts(cands, prev, partwts)]

            cands = [parts[matches[i]] for parts in cands]
            for parts in cands:
                self.refine_2way(parts, level, partwts, imbalance=imbalance)

        return self.best_parts(cands, graphs[0], partwts)

    def best_parts(self, cands, graph, partwts):
        scores = [self.score(parts, graph, partwts) for parts in cands]
        return cands[scores.index(min(scores))]

    def _bisect(self, graph, target_ratio, imbalance=None):
        nv = graph.nv
        target_wt = target_ratio*graph.vwts.sum(axis=0)
        partwts = np.array([target_ratio, 1 - target_ratio])
        cands = []

        for i in range(min(self.nstarts, nv)):
            if i == 0:
                seed = np.diff(graph.vtab).argmin()
                start = graph.bfs(seed).argmax()
            else:
                start = self.rng.integers(nv)

            parts = self.grow_bisection(graph, start, target_wt)
            self.refine_2way(parts, graph, partwts, imbalance=imbalance)

            # Only keep distinct bisections
            if not any(np.array_equal(parts, c) for c in cands):
                cands.append(parts)

        # Return the best few candidates
        cands.sort(key=lambda p: self.score(p, graph, partwts))
        return cands[:4]

    def refine_2way(self, parts, graph, partwts, check=False, imbalance=None):
        maximb = 1.2 if graph.ncon == 1 else 2

        for _ in range(self.nrefine):
            if not self._fm_2way(parts, graph, partwts, imbalance):
                break

        if not check:
            return

        # Reattach any islands and repair the balance if this disturbed it
        if self.eliminate_components(parts, graph, partwts, maximb):
            self._fm_2way(parts, graph, partwts, imbalance)
            self.eliminate_components(parts, graph, partwts)

    def _fm_2way(self, parts, graph, partwts, imbalance=None):
        if (nv := graph.nv) <= 2:
            return False

        targets, upper, _ = self._bounds(partwts, graph, imbalance)
        rtargets = np.maximum(targets, 1e-10)

        fm = TwoWayFMRefiner(graph, parts)
        return fm.refine(upper, rtargets, min(max(nv // 100, 15), 100))

    def grow_bisection(self, graph, start, target_wt):
        vtab, etab, vwts = graph.vtab, graph.etab, graph.vwts
        nv, nc = vwts.shape
        parts = np.ones(nv, dtype=np.int32)
        in_set = np.zeros(nv, dtype=bool)
        gain = np.zeros(nv, dtype=int)

        in_set[start] = True
        parts[start] = 0
        cur_wt = vwts[start].astype(float)

        # Initialize gains from start vertex's neighbors
        np.add.at(gain, etab[vtab[start]:vtab[start + 1]], 1)

        while True:
            # Only add vertices carrying a constraint we are still short of
            if not (need := cur_wt < target_wt).any():
                break

            adds = ~in_set & (vwts[:, need] > 0).any(axis=1)
            candidates = adds & (gain > 0)
            if not candidates.any():
                remaining = np.flatnonzero(adds)
                if len(remaining) == 0:
                    break
                v = remaining[self.rng.integers(len(remaining))]
            else:
                cidx = np.flatnonzero(candidates)
                cgains = gain[cidx]
                max_g = cgains.max()
                top = cidx[cgains == max_g]
                v = top[self.rng.integers(len(top))]

            if nc == 1 and cur_wt[0] + vwts[v, 0] > 1.05*target_wt[0]:
                break

            in_set[v] = True
            parts[v] = 0
            cur_wt += vwts[v]

            np.add.at(gain, etab[vtab[v]:vtab[v + 1]], 1)

        return parts

    def _greedy_refine(self, parts, graph, partwts):
        vtab, etab, ewts = graph.vtab, graph.etab, graph.ewts
        vwts, edge_src = graph.vwts, graph.esrc
        nv, nparts = graph.nv, len(partwts)

        _, upper, lower = self._bounds(partwts, graph)
        part_wts = graph.part_wts(parts, nparts)

        degrees = np.bincount(edge_src, ewts, minlength=nv)

        # Find boundary: vertices with some neighbours in other parts
        same = parts[edge_src] == parts[etab]
        own_cnt = np.bincount(edge_src, same*ewts, minlength=nv)
        boundary = np.flatnonzero(degrees > own_cnt)
        if len(boundary) == 0:
            return

        # Gather only boundary-adjacent edges via vtab offsets
        nb = len(boundary)
        b_eidx, b_deg = graph.incident(boundary)

        # Build nbr_cnt for boundary vertices only
        b_vidx = np.repeat(np.arange(nb), b_deg)
        b_nbr_p = parts[etab[b_eidx]]
        flat = b_vidx*nparts + b_nbr_p
        b_nbr = np.bincount(flat, ewts[b_eidx], minlength=nb*nparts)
        b_nbr = b_nbr.reshape(nb, nparts)

        b_src, b_dst, b_gain, _ = boundary_gains(b_nbr, parts, boundary)

        # Filter to positive-gain moves
        good = np.flatnonzero(b_gain > 0)
        order = good[np.argsort(b_gain[good])[::-1]]

        # Map each boundary vertex to its row of neighbour totals
        brow = np.full(nv, -1)
        brow[boundary] = np.arange(nb)

        for i in order.tolist():
            v, src, dst = int(boundary[i]), int(b_src[i]), int(b_dst[i])

            # Skip moves which earlier moves have made unprofitable
            if b_nbr[i, dst] <= b_nbr[i, src]:
                continue

            vw = vwts[v]
            if not can_move(part_wts, src, dst, vw, upper, lower):
                continue

            parts[v] = dst
            part_wts[src] -= vw
            part_wts[dst] += vw

            # Update the neighbour totals of the boundary neighbours
            eidx = slice(vtab[v], vtab[v + 1])
            rows, ew = brow[etab[eidx]], ewts[eidx]
            nbm = rows >= 0
            b_nbr[rows[nbm], src] -= ew[nbm]
            b_nbr[rows[nbm], dst] += ew[nbm]

    def refine_kway(self, parts, graph, partwts, nrefine, btol=0.02):
        if graph.nv <= len(partwts) or graph.ne == 0:
            return

        targets, upper, lower = self._bounds(partwts, graph)

        # Widen the bounds so that the heaviest vertex can always move
        if btol > 0:
            slack = graph.vwts.max(axis=0)
            upper = np.maximum(upper, targets + slack)
            lower = np.minimum(lower, targets - slack)

        # Ignore weight components absent from the graph
        rupper = np.where(upper > 0, upper, 1)

        ref = GreedyKWayRefiner(graph, parts, targets, 1.0 + self.imbalance)

        for _ in range(nrefine):
            # Phase 1: balance pass if needed
            nbal = 0
            for _ in range(nrefine):
                if (ref.part_wts / rupper).max() - 1 <= btol:
                    break

                if not (nmoved := ref.balance_pass(upper)):
                    break

                nbal += nmoved

            # Phase 2: FM pass with adaptive upper bounds
            balanced = (ref.part_wts / rupper).max() - 1 <= btol
            improved = ref.refine_pass(np.maximum(upper, ref.part_wts), lower)

            # Stop once neither the balance nor the cut is improving
            if not improved and (balanced or not nbal):
                break
