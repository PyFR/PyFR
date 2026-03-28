import numpy as np

from pyfr.nputil import iter_struct
from pyfr.partitioners.base import BasePartitioner


class BaselinePartitioner(BasePartitioner):
    name = 'baseline'
    has_part_weights = True
    has_multiple_constraints = True

    int_opts = {
        'seed', 'nrefine', 'ufactor', 'niter', 'nvcycles',
        'coarsen_min_mult', 'coarsen_min_abs', 'coarsen_stop_pct',
        'rb_limit', 'init_nparts_small', 'greedy_restarts_sc',
        'greedy_restarts_mc'
    }
    enum_opts = {}
    dflt_opts = {
        'seed': 2079, 'nrefine': 10, 'ufactor': 10, 'niter': 1,
        'nvcycles': 4, 'coarsen_min_mult': 20, 'coarsen_min_abs': 500,
        'coarsen_stop_pct': 80, 'rb_limit': 64, 'init_nparts_small': 16,
        'greedy_restarts_sc': 4, 'greedy_restarts_mc': 8
    }

    @staticmethod
    def _wbincount(keys, vwts, n):
        if vwts.ndim == 1:
            return np.bincount(keys, vwts, minlength=n)
        else:
            return np.column_stack(
                [np.bincount(keys, col, minlength=n) for col in vwts.T]
            )

    @staticmethod
    def _targets(partwts, vwts):
        if vwts.ndim == 1:
            return partwts*float(vwts.sum())
        else:
            return partwts[:, None]*vwts.sum(axis=0)

    def _bounds(self, partwts, vwts):
        ufactor = self.opts['ufactor'] / 1000.0
        targets = self._targets(partwts, vwts)
        upper = np.maximum(targets*(1 + ufactor), np.ceil(targets))
        lower = max(1 - ufactor, 0)*targets
        return targets, upper, lower

    @staticmethod
    def _boundary_gains(rows, parts, boundary, nv=0):
        nb = len(boundary)
        b_idx = np.arange(nb)
        b_src = parts[boundary]
        b_own = rows[b_idx, b_src].astype(np.int32)
        b_gains = rows.astype(np.int32) - b_own[:, None]
        b_gains[b_idx, b_src] = -(nv if nv else int(b_own.max()) + 1)
        b_dst = b_gains.argmax(axis=1)
        b_gain = b_gains[b_idx, b_dst]
        return b_src, b_dst, b_gain, b_own

    @staticmethod
    def _best_dest(row, src):
        saved = int(row[src])
        row[src] = -1
        dst = int(row.argmax())
        row[src] = saved
        return dst, saved

    @staticmethod
    def _can_move(part_wts, src, dst, vw, upper, lower):
        return not ((part_wts[dst] + vw > upper[dst]).any() or
                    (part_wts[src] - vw < lower[src]).any())

    @staticmethod
    def _rollback_moves(parts, vtab, etab, nbr_cnt, moves, best_idx):
        for v, src, dst in reversed(moves[best_idx + 1:]):
            parts[v] = src
            nbs = etab[vtab[v]:vtab[v + 1]]
            nbr_cnt[nbs, dst] -= 1
            nbr_cnt[nbs, src] += 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rng = np.random.default_rng(self.opts['seed'])

    def _partition_graph(self, graph, partwts):
        nv = len(graph.vtab) - 1
        self._idxdtype = idt = np.int32 if nv < 2**31 else np.int64

        vtab = graph.vtab.astype(idt)
        etab = graph.etab.astype(idt)
        vwts = np.ascontiguousarray(graph.vwts.squeeze(), dtype=np.int32)

        partwts = np.asarray(partwts, dtype=float)
        partwts /= partwts.sum()

        nparts = len(partwts)
        if nparts == 1:
            return np.zeros(nv, dtype=np.int32)

        # Precompute edge sources
        edge_src = np.repeat(np.arange(nv, dtype=idt), np.diff(vtab))

        return self._multilevel_kway(vtab, etab, vwts, partwts, edge_src)

    def _multilevel_kway(self, vtab, etab, vwts, partwts, edge_src):
        nv, nparts = len(vtab) - 1, len(partwts)

        if nv <= 1:
            return np.zeros(nv, dtype=np.int32)

        # Build coarsening hierarchy
        ewts = np.ones(len(etab), dtype=np.int32)
        hierarchy = [(vtab, etab, ewts, vwts, None, edge_src)]
        min_size = max(self.opts['coarsen_min_mult']*nparts,
                       self.opts['coarsen_min_abs'])
        cstop = self.opts['coarsen_stop_pct'] / 100.0

        while True:
            c_vtab, c_etab, c_ewts, c_vwts, _, c_esrc = hierarchy[-1]
            if len(c_vtab) - 1 <= min_size:
                break

            result = self._coarsen(c_vtab, c_etab, c_ewts, c_vwts, c_esrc)
            if (result is None or
                len(result[0]) - 1 >= cstop*(len(c_vtab) - 1)):
                break

            hierarchy.append(result)

        # Initial k-way partition at coarsest level
        c_vtab, c_etab, _, c_vwts, _, c_esrc = hierarchy[-1]
        if c_vwts.ndim > 1:
            c_targets = self._targets(partwts, c_vwts)
        else:
            c_targets = None

        def score(p):
            cut = (p[c_esrc] != p[c_etab]).sum()
            if c_targets is not None:
                pw = self._wbincount(p, c_vwts, nparts)
                return ((pw / np.maximum(c_targets, 1e-10)).max(), cut)
            else:
                return (0, cut)

        parts = self._greedy_kway(c_vtab, c_etab, c_vwts, partwts, c_esrc)
        best_score = score(parts)

        if nparts <= self.opts['rb_limit']:
            rb_parts = self._rb_kway(c_vtab, c_etab, c_vwts, partwts)
            if score(rb_parts) < best_score:
                parts = rb_parts

        # Uncoarsen with k-way refinement
        nlevels = len(hierarchy) - 1
        for i in range(nlevels - 1, -1, -1):
            f_vtab, f_etab, _, f_vwts, _, f_esrc = hierarchy[i]
            _, _, _, _, match, _ = hierarchy[i + 1]

            parts = parts[match]

            self._greedy_refine(parts, f_vtab, f_etab, f_vwts, partwts, f_esrc)

            if self.opts['nrefine'] > 0:
                self._refine_kway(parts, f_vtab, f_etab, f_vwts, partwts,
                                  f_esrc)

        # Apply multigrid
        if len(hierarchy) > 2:
            vc_nrefine = min(self.opts['nrefine'], 5)
            nvcycles = self.opts['nvcycles']

            f_vtab, f_etab, _, f_vwts, _, f_esrc = hierarchy[0]
            _, f_upper, _ = self._bounds(partwts, f_vwts)

            def vc_score(p):
                cut = (p[f_esrc] != p[f_etab]).sum()
                if f_vwts.ndim > 1:
                    pw = self._wbincount(p, f_vwts, nparts)
                    excess = max(0, (pw - f_upper).max())
                    return (excess, cut)
                else:
                    return (0, cut)

            best_score = vc_score(parts)
            best_parts = parts.copy()
            no_improve = 0

            for _ in range(nvcycles):
                # Project up to coarsest level
                for i in range(1, len(hierarchy)):
                    c_vtab, _, _, _, match, _ = hierarchy[i]
                    c_parts = np.zeros(len(c_vtab) - 1, dtype=np.int32)
                    c_parts[match] = parts
                    parts = c_parts

                # Refine back down to finest level
                for i in range(len(hierarchy) - 2, -1, -1):
                    lv_vtab, lv_etab, _, lv_vwts, _, lv_esrc = hierarchy[i]
                    _, _, _, _, match, _ = hierarchy[i + 1]

                    parts = parts[match]

                    self._greedy_refine(parts, lv_vtab, lv_etab, lv_vwts,
                                        partwts, lv_esrc)

                    if vc_nrefine > 0:
                        self._refine_kway(parts, lv_vtab, lv_etab, lv_vwts,
                                          partwts, lv_esrc, nrefine=vc_nrefine)

                # Track best and stop after 2 consecutive misses
                cur_score = vc_score(parts)
                if cur_score < best_score:
                    best_score = cur_score
                    best_parts = parts.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= 2:
                        break

            parts[:] = best_parts

        return parts

    def _greedy_kway(self, vtab, etab, vwts, partwts, edge_src):
        nv, nparts = len(vtab) - 1, len(partwts)

        if nv <= nparts:
            return np.arange(nv, dtype=np.int32) % nparts

        targets = self._targets(partwts, vwts)
        mc = vwts.ndim > 1

        # Multiple restarts
        nrestarts = self.opts[f'greedy_restarts_{"mc" if mc else "sc"}']
        if nparts > self.opts['init_nparts_small']:
            nrestarts = max(1, nrestarts // 2)

        best = (np.inf, np.inf, None)
        for attempt in range(nrestarts):
            seeds = self._pick_seeds(vtab, etab, nparts, attempt)
            parts = self._grow_kway(vtab, etab, vwts, targets, seeds)

            cut = (parts[edge_src] != parts[etab]).sum()
            pw = self._wbincount(parts, vwts, nparts)
            imb = (pw / np.maximum(targets, 1e-10)).max()
            score = (imb, cut)

            if score < best[:2]:
                best = (*score, parts)

        return best[2]

    def _rb_kway(self, vtab, etab, vwts, partwts):
        nv, nparts = len(vtab) - 1, len(partwts)

        if nparts == 1:
            return np.zeros(nv, dtype=np.int32)

        if nv <= nparts:
            return np.arange(nv, dtype=np.int32) % nparts

        # Split partition weights into two balanced groups
        pw = np.asarray(partwts)
        half = pw.sum() / 2
        cumwt = np.cumsum(pw)
        split = int(np.searchsorted(cumwt, half, side='right'))
        split = max(1, min(split, nparts - 1))
        left_wt = pw[:split].sum()
        ratio = left_wt / pw.sum()

        # Bisect the graph
        bparts = self._bisect(vtab, etab, vwts, ratio)

        # Map partition 0 -> left group, partition 1 -> right group
        parts = np.empty(nv, dtype=np.int32)

        for side, prange in [(0, range(split)), (1, range(split, nparts))]:
            vidx = np.flatnonzero(bparts == side)
            npk = len(prange)

            if npk == 1:
                parts[vidx] = prange.start
            else:
                sv, se = self._extract_subgraph(vidx, vtab, etab)
                sub_vwts = vwts[vidx]
                sub_pw = pw[prange.start:prange.stop]
                sub_parts = self._rb_kway(sv, se, sub_vwts, sub_pw)

                # Remap local partition IDs to global
                parts[vidx] = sub_parts + prange.start

        return parts

    def _pick_seeds(self, vtab, etab, k, attempt=0):
        nv = len(vtab) - 1
        if k >= nv:
            return np.arange(k, dtype=int) % nv

        if attempt == 0:
            start = np.diff(vtab).argmin()
        else:
            start = self._rng.integers(nv)

        # BFS from start to find a peripheral vertex
        dist = self._bfs(start, vtab, etab)
        dist[dist < 0] = 0
        seeds = [int(dist.argmax())]

        # Greedily pick farthest vertices from existing seeds
        min_dist = self._bfs(seeds[0], vtab, etab)
        min_dist[min_dist < 0] = nv

        for _ in range(k - 1):
            next_seed = int(min_dist.argmax())
            seeds.append(next_seed)
            d = self._bfs(next_seed, vtab, etab)
            d[d < 0] = nv
            np.minimum(min_dist, d, out=min_dist)

        return np.array(seeds)

    def _grow_kway(self, vtab, etab, vwts, targets, seeds):
        nv, nparts = len(vtab) - 1, len(seeds)

        if vwts.ndim == 1:
            vwts = vwts.reshape(-1, 1)
            targets = targets.reshape(-1, 1)

        nc = vwts.shape[1]
        targets = np.asarray(targets, dtype=float)

        parts = np.full(nv, -1, dtype=np.int32)
        part_wts = np.zeros((nparts, nc), dtype=vwts.dtype)
        avail = bytearray(b'\x01'*nv)

        # Number of edges from v into partition p
        gain = [[0]*nparts for _ in range(nv)]

        # Per-partition candidate sets (vertices with gain > 0)
        cands = [set() for _ in range(nparts)]

        for p, s in enumerate(seeds):
            s = int(s)
            parts[s] = p
            avail[s] = 0
            part_wts[p] = vwts[s]

            for nb in iter_struct(etab[vtab[s]:vtab[s + 1]]):
                gain[nb][p] += 1
                cands[p].add(nb)

        unassigned = nv - nparts
        rng = self._rng

        while unassigned > 0:
            moved = False

            for p in range(nparts):
                if np.all(part_wts[p] >= targets[p]):
                    continue

                # Scan candidates for max gain, pruning stale
                best_g, best_vs, stale = 0, [], []
                for v in cands[p]:
                    if not avail[v]:
                        stale.append(v)
                    elif (g := gain[v][p]) > best_g:
                        best_g, best_vs = g, [v]
                    elif g == best_g and g > 0:
                        best_vs.append(v)

                cands[p].difference_update(stale)
                if not best_vs:
                    continue

                v = best_vs[int(rng.integers(len(best_vs)))]

                parts[v] = p
                avail[v] = 0
                part_wts[p] += vwts[v]
                unassigned -= 1
                cands[p].discard(v)
                moved = True

                for nb in iter_struct(etab[vtab[v]:vtab[v + 1]]):
                    if avail[nb]:
                        gain[nb][p] += 1
                        cands[p].add(nb)

            # Fallback for disconnected vertices
            if not moved:
                remaining = np.flatnonzero(parts < 0)
                if len(remaining) == 0:
                    break

                imb = part_wts / np.maximum(targets, 1e-10)
                lightest = int(imb.max(axis=1).argmin())
                v = int(remaining[rng.integers(len(remaining))])
                parts[v] = lightest
                avail[v] = 0
                part_wts[lightest] += vwts[v]
                unassigned -= 1

                for nb in iter_struct(etab[vtab[v]:vtab[v + 1]]):
                    if avail[nb]:
                        gain[nb][lightest] += 1
                        cands[lightest].add(nb)

        return parts

    def _extract_subgraph(self, vidx, vtab, etab):
        idt = self._idxdtype
        nv_sub = len(vidx)
        if not nv_sub:
            return np.array([0]), np.array([], dtype=idt)

        # Map global to local indices
        gmap = np.full(len(vtab) - 1, -1, dtype=idt)
        gmap[vidx] = np.arange(nv_sub, dtype=idt)

        # Get edge ranges
        starts, ends = vtab[vidx], vtab[vidx + 1]
        counts = ends - starts
        total = counts.sum()

        if total == 0:
            return np.zeros(nv_sub + 1, dtype=int), np.array([], dtype=idt)

        # Vectorized gather of all neighbours
        v_idx = np.repeat(np.arange(nv_sub), counts)
        base = np.repeat(np.r_[0, counts[:-1].cumsum()], counts)
        all_nb = etab[starts[v_idx] + np.arange(total) - base]

        # Filter to internal edges
        local_nb = gmap[all_nb]
        valid = local_nb >= 0

        # Build sub CSR
        valid_counts = np.bincount(v_idx[valid], minlength=nv_sub)

        sub_vtab = np.zeros(nv_sub + 1, dtype=int)
        sub_vtab[1:] = valid_counts.cumsum()
        sub_etab = local_nb[valid]

        return sub_vtab, sub_etab

    def _coarsen(self, vtab, etab, ewts, vwts, edge_src):
        nv = len(vtab) - 1
        if nv <= 1:
            return None

        if len(etab) == 0:
            return (vtab.copy(), etab.copy(), ewts.copy(), vwts.copy(),
                    np.arange(nv, dtype=self._idxdtype), edge_src.copy())

        # Heavy edge matching
        match = self._heavy_edge_match(vtab, etab, edge_src, ewts)
        nv_c = match.max() + 1

        # Coarse vertex weights
        c_vwts = self._wbincount(match, vwts, nv_c).astype(np.int32)

        # Map edges to coarse vertices and remove self-loops
        c_src = match[edge_src]
        c_dst = match[etab]
        keep = c_src != c_dst

        if not keep.any():
            idt = self._idxdtype
            return (np.zeros(nv_c + 1, dtype=int), np.array([], dtype=idt),
                    np.array([], dtype=np.int32), c_vwts, match,
                    np.array([], dtype=idt))

        c_src, c_dst, c_ewts = c_src[keep], c_dst[keep], ewts[keep]

        # Sum weights of parallel edges via sort-and-deduplicate
        edge_id = c_src.astype(int, copy=False)*nv_c + c_dst
        order = np.argsort(edge_id)
        edge_id, c_ewts = edge_id[order], c_ewts[order]

        diff = np.empty(len(edge_id), dtype=bool)
        diff[0] = True
        np.not_equal(edge_id[1:], edge_id[:-1], out=diff[1:])

        # Accumulate weights for duplicate edges
        labels = np.cumsum(diff) - 1
        c_ewts = np.bincount(labels, c_ewts, minlength=diff.sum())
        c_ewts = c_ewts.astype(np.int32)

        edge_id = edge_id[diff]
        c_src, c_dst = np.divmod(edge_id, nv_c)
        c_dst = c_dst.astype(self._idxdtype)

        # Build coarse CSR
        c_vtab = np.zeros(nv_c + 1, dtype=int)
        c_vtab[1:] = np.bincount(c_src, minlength=nv_c).cumsum()

        c_esrc = np.repeat(np.arange(nv_c, dtype=self._idxdtype),
                           np.diff(c_vtab))

        return c_vtab, c_dst, c_ewts, c_vwts, match, c_esrc

    def _heavy_edge_match(self, vtab, etab, edge_src, ewts=None):
        nv, ne = len(vtab) - 1, len(etab)
        idt = self._idxdtype

        if ne == 0:
            return np.arange(nv, dtype=idt)

        partner = np.full(nv, -1, dtype=idt)
        arange_nv = np.arange(nv)

        # KaHIP-style inner/outer edge rating: ewt / (deg_u * deg_v)
        # favours edges between low-degree vertices, encouraging
        # merging of tightly-coupled clusters
        degrees = np.diff(vtab).astype(float)
        deg_prod = degrees[edge_src]*degrees[etab]
        base_scores = np.empty(ne, dtype=float)
        base_scores[:] = ewts if ewts is not None else 1.0
        np.divide(base_scores, deg_prod, out=base_scores, where=deg_prod > 0)
        noise_scale = 0.5*base_scores.max()

        # Preallocate per-round arrays to avoid repeated allocation
        scores = np.empty(ne, dtype=float)
        best_score = np.empty(nv, dtype=float)
        proposal = np.empty(nv, dtype=idt)
        has_edges = np.diff(vtab) > 0

        for _ in range(8):
            unmatched = partner < 0
            if unmatched.sum() <= 1:
                break

            um_mask = unmatched[edge_src] & unmatched[etab]
            if not um_mask.any():
                break

            # Score edges with random tiebreaker
            scores[:] = base_scores
            scores += self._rng.random(ne)*noise_scale
            scores[~um_mask] = -np.inf

            # Each unmatched vertex proposes to best unmatched neighbor;
            # reduceat exploits CSR order (edge_src is sorted)
            best_score[:] = -np.inf
            seg_max = np.maximum.reduceat(scores, vtab[:-1])
            best_score[has_edges] = seg_max[has_edges]

            is_best = um_mask & (scores >= best_score[edge_src] - 1e-10)
            best_eidx = np.flatnonzero(is_best)
            if len(best_eidx) == 0:
                break

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
            one_sided = (proposal >= 0) & ~mutual & (partner < 0)
            if one_sided.any():
                proposers = np.flatnonzero(one_sided)
                targets = proposal[proposers]
                keep = partner[targets] < 0
                proposers, targets = proposers[keep], targets[keep]

                if len(targets):
                    _, idx = np.unique(targets, return_index=True)
                    proposers, targets = proposers[idx], targets[idx]

                    # A vertex that is both a proposer and a target
                    # cannot serve both roles in vectorized assignment;
                    # drop the entry where it appears as a target
                    overlap = np.isin(targets, proposers)
                    if overlap.any():
                        keep = ~overlap
                        proposers = proposers[keep]
                        targets = targets[keep]

                    if len(proposers):
                        partner[proposers] = targets
                        partner[targets] = proposers

        # Build coarse vertex ID mapping from matched pairs
        is_owner = (partner >= 0) & (arange_nv < partner)
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

    def _bisect(self, vtab, etab, vwts, target_ratio):
        nv = len(vtab) - 1
        if nv <= 1:
            return np.zeros(nv, dtype=np.int32)

        # Scalar weights for growth heuristic; for multi-constraint,
        # scale each constraint by the inverse of its total weight so
        # all constraints contribute equally to the growth balance
        mc = vwts.ndim > 1
        if mc:
            totals = vwts.sum(axis=0).astype(float)
            totals = np.maximum(totals, 1.0)
            scale = totals.max() / totals
            svwts = (vwts * scale).sum(axis=1)
        else:
            svwts = vwts

        target_wt = target_ratio*svwts.sum()
        edge_src = np.repeat(np.arange(nv, dtype=self._idxdtype),
                             np.diff(vtab))
        nstarts = max(4, self.opts['niter'])
        partwts = np.array([target_ratio, 1 - target_ratio])
        best_parts = None
        best_score = (np.inf, np.inf)

        for i in range(min(nstarts, max(1, nv // 10))):
            if i == 0:
                seed = np.diff(vtab).argmin()
                start = self._bfs(seed, vtab, etab).argmax()
            else:
                start = self._rng.integers(nv)

            parts = self._bisect_ggp(start, vtab, etab, svwts, target_wt)

            # For multi-constraint: fix per-constraint balance via
            # vertex swaps before FM refinement
            if mc:
                self._bisect_balance(parts, vwts, partwts)

            self._refine_kway(parts, vtab, etab, vwts, partwts, edge_src,
                              nrefine=5)

            cut = (parts[edge_src] != parts[etab]).sum()
            if mc:
                pw = self._wbincount(parts, vwts, 2)
                _, ub2, _ = self._bounds(partwts, vwts)
                excess = max(0, (pw - ub2).max())
                score = (excess, cut)
            else:
                score = (0, cut)

            if score < best_score:
                best_score = score
                best_parts = parts.copy()

        return best_parts

    def _bisect_balance(self, parts, vwts, partwts):
        nc = vwts.shape[1]
        total_wt = vwts.sum(axis=0).astype(float)
        targets = partwts[:, None]*total_wt
        svwts = vwts.sum(axis=1)

        for _ in range(nc*8):
            part_wts = self._wbincount(parts, vwts, 2)
            imb = part_wts / np.maximum(targets, 1e-10)

            # Find the most violated constraint
            col_max = imb.max(axis=0)
            worst_c = int(col_max.argmax())
            worst_imb = col_max[worst_c]

            if worst_imb <= 1.03:
                break

            c = worst_c
            oside = int(imb[:, c].argmax())
            uside = 1 - oside

            # On overweight side: high fraction of weight in c
            on_over = np.flatnonzero(parts == oside)
            on_under = np.flatnonzero(parts == uside)

            if len(on_over) == 0 or len(on_under) == 0:
                break

            ov_frac = vwts[on_over, c] / np.maximum(svwts[on_over], 1)
            un_frac = vwts[on_under, c] / np.maximum(svwts[on_under], 1)

            # Sort: overweight by desc c-fraction, under by asc
            ov_order = on_over[np.argsort(ov_frac)[::-1]]
            un_order = on_under[np.argsort(un_frac)]

            swapped = False
            ui = 0
            for ov_v in ov_order:
                ov_v = int(ov_v)
                ow = vwts[ov_v].astype(float)
                if ow[c] == 0:
                    break

                # Find a partner on the underweight side with
                # similar total weight but less c-weight
                while ui < len(un_order):
                    un_v = int(un_order[ui])
                    uw = vwts[un_v].astype(float)

                    # Net transfer: ow goes to under, uw goes to over
                    new_over = part_wts[oside] - ow + uw
                    new_under = part_wts[uside] + ow - uw

                    ov_imb = new_over / np.maximum(targets[oside], 1e-10)
                    un_imb = new_under / np.maximum(targets[uside], 1e-10)
                    new_imb = np.maximum(ov_imb, un_imb).max()

                    if new_imb < worst_imb:
                        parts[ov_v] = uside
                        parts[un_v] = oside
                        part_wts[oside] = new_over
                        part_wts[uside] = new_under
                        swapped = True
                        ui += 1
                        break

                    ui += 1

                if ui >= len(un_order):
                    break

            if not swapped:
                break

    def _bisect_ggp(self, start, vtab, etab, vwts, target_wt):
        nv = len(vtab) - 1
        parts = np.ones(nv, dtype=np.int32)
        in_set = np.zeros(nv, dtype=bool)
        gain = np.zeros(nv, dtype=int)

        in_set[start] = True
        parts[start] = 0
        cur_wt = vwts[start]

        # Initialize gains from start vertex's neighbors
        np.add.at(gain, etab[vtab[start]:vtab[start + 1]], 1)

        while cur_wt < target_wt:
            candidates = ~in_set & (gain > 0)
            if not candidates.any():
                remaining = np.flatnonzero(~in_set)
                if len(remaining) == 0:
                    break
                v = remaining[self._rng.integers(len(remaining))]
            else:
                cidx = np.flatnonzero(candidates)
                cgains = gain[cidx]
                max_g = cgains.max()
                top = cidx[cgains == max_g]
                v = top[self._rng.integers(len(top))]

            if cur_wt + vwts[v] > target_wt*1.05:
                break

            in_set[v] = True
            parts[v] = 0
            cur_wt += vwts[v]

            np.add.at(gain, etab[vtab[v]:vtab[v + 1]], 1)

        return parts

    def _bfs(self, start, vtab, etab):
        nv = len(vtab) - 1
        levels = np.full(nv, -1, dtype=np.int32)
        levels[start] = 0

        frontier = np.array([start], dtype=self._idxdtype)
        level = 1

        while len(frontier) > 0:
            starts, ends = vtab[frontier], vtab[frontier + 1]
            counts = ends - starts
            total = counts.sum()

            if total == 0:
                break

            # Vectorized neighbour gather
            idx = np.repeat(np.arange(len(frontier)), counts)
            base = np.repeat(np.r_[0, counts[:-1].cumsum()], counts)
            neighbours = etab[starts[idx] + np.arange(total) - base]

            # Filter to unvisited
            unvisited_mask = levels[neighbours] < 0
            if not unvisited_mask.any():
                break

            unvisited = neighbours[unvisited_mask]
            levels[unvisited] = level

            # Deduplicate frontier via bincount
            seen = np.bincount(unvisited, minlength=nv)
            frontier = np.flatnonzero(seen)
            level += 1

        return levels

    def _greedy_refine(self, parts, vtab, etab, vwts, partwts, edge_src):
        nv, nparts = len(vtab) - 1, len(partwts)

        if nv <= nparts or len(etab) == 0:
            return

        _, upper, lower = self._bounds(partwts, vwts)
        part_wts = self._wbincount(parts, vwts, nparts)

        degrees = np.diff(vtab)

        # Find boundary: vertices with some neighbours in other parts
        same = parts[edge_src] == parts[etab]
        own_cnt = np.bincount(edge_src, same, minlength=nv)
        boundary = np.flatnonzero((degrees > own_cnt) & (degrees > 0))
        if len(boundary) == 0:
            return

        # Gather only boundary-adjacent edges via vtab offsets
        nb = len(boundary)
        b_deg = degrees[boundary]
        total_be = b_deg.sum()
        offsets = vtab[boundary] - np.r_[0, b_deg.cumsum()[:-1]]
        b_eidx = np.repeat(offsets, b_deg) + np.arange(total_be)

        # Build nbr_cnt for boundary vertices only
        b_vidx = np.repeat(np.arange(nb), b_deg)
        b_nbr_p = parts[etab[b_eidx]]
        flat = b_vidx*nparts + b_nbr_p
        b_nbr = np.bincount(flat, minlength=nb*nparts)
        b_nbr = b_nbr.reshape(nb, nparts)

        b_src, b_dst, b_gain, _ = self._boundary_gains(b_nbr, parts, boundary)

        # Filter to positive-gain moves
        good = np.flatnonzero(b_gain > 0)
        if len(good) == 0:
            return

        order = good[np.argsort(b_gain[good])[::-1]]

        for v, src, dst in zip(boundary[order], b_src[order], b_dst[order]):
            v, src, dst = int(v), int(src), int(dst)
            vw = vwts[v]
            if ((part_wts[dst] + vw > upper[dst]).any() or
                (part_wts[src] - vw < lower[src]).any()):
                continue

            parts[v] = dst
            part_wts[src] -= vw
            part_wts[dst] += vw

    def _mc_greedy_pass(self, parts, vtab, etab, vwts, nbr_cnt, bm, ubvec,
                        minpwgts, maxpwgts, degrees, part_wts, omode):
        own_cnt = np.take_along_axis(nbr_cnt, parts[:, None], axis=1).ravel()
        boundary = np.flatnonzero(degrees - own_cnt > 0)

        if len(boundary) == 0:
            return 0

        b_src, b_dst, b_gain, b_own = self._boundary_gains(
            nbr_cnt[boundary], parts, boundary
        )

        # For balance mode: accept if dst has any edges; else gain >= 0
        if omode == 'balance':
            good = np.flatnonzero(b_gain + b_own > 0)
        else:
            good = np.flatnonzero(b_gain >= 0)

        if len(good) == 0:
            return 0

        # Sort candidates by gain descending
        order = good[np.argsort(b_gain[good])[::-1]]

        nmoved = 0
        for v, src, dst in zip(boundary[order], b_src[order], b_dst[order]):
            v, src, dst = int(v), int(src), int(dst)
            vw = vwts[v]

            # Check source lower bound
            if (part_wts[src] - vw < minpwgts[src]).any():
                continue

            # Check destination upper bound
            if (part_wts[dst] + vw > maxpwgts[dst]).any():
                if omode == 'refine':
                    continue

                # Balance mode: accept if the move improves the
                # overall balance (max excess then L2 norm)
                src_e = bm[src]*(part_wts[src] - vw) - ubvec
                dst_e = bm[dst]*(part_wts[dst] + vw) - ubvec
                src_n = bm[src]*part_wts[src] - ubvec
                dst_n = bm[dst]*part_wts[dst] - ubvec
                new_max = max(src_e.max(), dst_e.max())
                old_max = max(src_n.max(), dst_n.max())

                if new_max > old_max:
                    continue
                elif new_max == old_max:
                    new_l2 = (src_e*src_e).sum() + (dst_e*dst_e).sum()
                    old_l2 = (src_n*src_n).sum() + (dst_n*dst_n).sum()
                    if new_l2 >= old_l2:
                        continue

            parts[v] = dst
            part_wts[src] -= vw
            part_wts[dst] += vw
            nmoved += 1

            nbs = etab[vtab[v]:vtab[v + 1]]
            nbr_cnt[nbs, src] -= 1
            nbr_cnt[nbs, dst] += 1

        return nmoved

    def _refine_kway(self, parts, vtab, etab, vwts, partwts, edge_src,
                     nrefine=None):
        nv, nparts = len(vtab) - 1, len(partwts)
        if nv <= nparts or len(etab) == 0:
            return

        if nrefine is None: nrefine = self.opts['nrefine']
        targets, upper, lower = self._bounds(partwts, vwts)
        ufactor = self.opts['ufactor'] / 1000.0

        # Precompute reusable arrays for this level
        degrees = np.diff(vtab)
        moved = np.zeros(nv, dtype=bool)

        # nbr_cnt[v, p] = number of edges from v to partition p;
        # use int8 when max degree allows, else int16.
        parts_etab = parts[etab]
        nbr_dtype = np.int8 if degrees.max() <= 127 else np.int16
        nbr_cnt = np.zeros((nv, nparts), dtype=nbr_dtype)
        flat = edge_src*nparts + parts_etab
        cnt = np.bincount(flat, minlength=nv*nparts)
        nbr_cnt[:] = cnt.reshape(nv, nparts)

        if vwts.ndim > 1:
            bm = 1.0 / np.maximum(targets, 1e-10)
            ubvec = np.full(vwts.shape[1], 1.0 + ufactor)
            minpwgts = (0.2*targets).astype(int)

            for _ in range(nrefine):
                part_wts = self._wbincount(parts, vwts, nparts)

                # Phase 1: balance pass if needed
                imb_diff = (part_wts*bm - ubvec[None, :]).max()
                if imb_diff > 0.02:
                    maxpwgts_b = targets*ubvec[None, :]
                    self._mc_greedy_pass(parts, vtab, etab, vwts, nbr_cnt, bm,
                                         ubvec, minpwgts, maxpwgts_b,
                                         degrees, part_wts, 'balance')

                # Phase 2: FM pass with adaptive upper bounds
                pw = self._wbincount(parts, vwts, nparts)
                imb_per_c = (pw*bm).max(axis=0)
                r_ub = np.maximum(1.0 + ufactor, imb_per_c)
                mc_ub = targets*r_ub[None, :]
                mc_upper = np.maximum(mc_ub, np.ceil(targets))

                moved[:] = False
                improved = self._fm_pass_edge_cut(parts, vtab, etab, vwts,
                                                  nbr_cnt, mc_upper, lower,
                                                  nparts, degrees, moved)
                if not improved and imb_diff <= 0.02:
                    break
        else:
            for _ in range(nrefine):
                moved[:] = False
                improved = self._fm_pass_edge_cut(parts, vtab, etab, vwts,
                                                  nbr_cnt, upper, lower,
                                                  nparts, degrees, moved)
                if not improved:
                    break

    def _fm_pass_edge_cut(self, parts, vtab, etab, vwts, nbr_cnt, upper, lower,
                          nparts, degrees, moved):
        nv = len(vtab) - 1
        part_wts = self._wbincount(parts, vwts, nparts)

        # Identify boundary vertices (some neighbours in other parts)
        own_cnt = np.take_along_axis(nbr_cnt, parts[:, None], axis=1).ravel()
        boundary = np.flatnonzero((degrees > own_cnt) & (degrees > 0))
        if len(boundary) == 0:
            return False

        _, _, b_best_gain, _ = self._boundary_gains(
            nbr_cnt[boundary], parts, boundary, nv
        )

        order = np.argsort(b_best_gain)[::-1]

        moves = []
        cum_gain = best_cum_gain = 0
        best_idx = -1
        max_no_improve = max(int(len(boundary)**0.5), nparts)
        no_improve = 0

        for v in iter_struct(boundary[order]):
            if moved[v]:
                continue

            src = int(parts[v])
            row = nbr_cnt[v]
            dst, own = self._best_dest(row, src)
            g = int(row[dst] - own)

            if g <= -2:
                continue

            vw = vwts[v]
            if not self._can_move(part_wts, src, dst, vw, upper, lower):
                continue

            parts[v] = dst
            moved[v] = True
            part_wts[src] -= vw
            part_wts[dst] += vw
            cum_gain += g

            moves.append((v, src, dst))
            if cum_gain > best_cum_gain:
                best_cum_gain = cum_gain
                best_idx = len(moves) - 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

            nbs = etab[vtab[v]:vtab[v + 1]]
            nbr_cnt[nbs, src] -= 1
            nbr_cnt[nbs, dst] += 1

        # Rollback moves after the best prefix
        self._rollback_moves(parts, vtab, etab, nbr_cnt, moves, best_idx)

        return best_cum_gain > 0
