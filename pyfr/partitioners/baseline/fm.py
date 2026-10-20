import heapq

import numpy as np

from pyfr.nputil import iter_struct


def boundary_gains(rows, parts, boundary):
    nb = len(boundary)
    b_idx = np.arange(nb)
    b_src = parts[boundary]
    b_own = rows[b_idx, b_src].astype(np.int32)
    b_gains = rows.astype(np.int32) - b_own[:, None]
    b_gains[b_idx, b_src] = -(int(b_own.max()) + 1)
    b_dst = b_gains.argmax(axis=1)
    b_gain = b_gains[b_idx, b_dst]
    return b_src, b_dst, b_gain, b_own


def can_move(part_wts, src, dst, vw, upper, lower):
    return not ((part_wts[dst] + vw > upper[dst]).any() or
                (part_wts[src] - vw < lower[src]).any())


class BaseRefiner:
    def __init__(self, graph, parts, nparts):
        nv = graph.nv
        self.vtab, self.etab, self.ewts = graph.vtab, graph.etab, graph.ewts
        self.vwts, self.parts, self.nparts = graph.vwts, parts, nparts
        self.part_wts = graph.part_wts(parts, nparts)

        # Precompute reusable arrays for this level
        degrees = np.bincount(graph.esrc, graph.ewts, minlength=nv)
        self.degrees = degrees.astype(np.int64)

        # Edge weight from each vertex into each partition, in a compact dtype
        parts_etab = parts[graph.etab]
        nbr_dtype = np.int8 if self.degrees.max() <= 127 else np.int32
        self.nbr_cnt = np.zeros((nv, nparts), dtype=nbr_dtype)
        flat = graph.esrc.astype(np.int64, copy=False)*nparts + parts_etab
        cnt = np.bincount(flat, graph.ewts, minlength=nv*nparts)
        self.nbr_cnt[:] = cnt.reshape(nv, nparts)


class GreedyKWayRefiner(BaseRefiner):
    def __init__(self, graph, parts, targets, ubvec):
        super().__init__(graph, parts, len(targets))

        self.bm = 1.0 / np.maximum(targets, 1e-10)
        self.ubvec = ubvec
        self.minpwgts = (0.2*targets).astype(int)

    def move(self, v, src, dst):
        vw = self.vwts[v]
        self.parts[v] = dst
        self.part_wts[src] -= vw
        self.part_wts[dst] += vw

        # Move the edges of v in the neighbour totals from src to dst
        eidx = slice(self.vtab[v], self.vtab[v + 1])
        nbrs, ew = self.etab[eidx], self.ewts[eidx]
        self.nbr_cnt[nbrs, src] -= ew
        self.nbr_cnt[nbrs, dst] += ew

    def rollback(self, moves, best_idx):
        for v, src, dst in reversed(moves[best_idx + 1:]):
            self.move(v, dst, src)

    @staticmethod
    def _best_dest(row, src):
        saved = int(row[src])
        row[src] = -1
        dst = int(row.argmax())
        row[src] = saved
        return dst, saved

    def balance_pass(self, maxpwgts):
        parts, nbr_cnt, part_wts = self.parts, self.nbr_cnt, self.part_wts
        bm, ubvec, minpwgts = self.bm, self.ubvec, self.minpwgts

        own_cnt = np.take_along_axis(nbr_cnt, parts[:, None], axis=1).ravel()
        boundary = np.flatnonzero(self.degrees - own_cnt > 0)

        # Restrict the boundary to vertices of overweight partitions
        over = (part_wts > maxpwgts).any(axis=1)
        boundary = boundary[over[parts[boundary]]]
        if len(boundary) == 0:
            return 0

        b_src, b_dst, b_gain, b_own = boundary_gains(
            nbr_cnt[boundary], parts, boundary
        )

        # For balance mode: accept if dst has any edges; else gain >= 0
        good = np.flatnonzero(b_gain + b_own > 0)

        # Sort candidates by gain descending
        order = good[np.argsort(b_gain[good])[::-1]]

        nmoved = 0
        for v, src, dst in zip(boundary[order], b_src[order], b_dst[order]):
            v, src, dst = int(v), int(src), int(dst)
            vw = self.vwts[v]

            # Check source lower bound
            if (part_wts[src] - vw < minpwgts[src]).any():
                continue

            # Check destination upper bound
            if (part_wts[dst] + vw > maxpwgts[dst]).any():
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

            self.move(v, src, dst)
            nmoved += 1

        return nmoved

    def refine_pass(self, upper, lower):
        parts, nbr_cnt, part_wts = self.parts, self.nbr_cnt, self.part_wts

        # Identify boundary vertices (some neighbours in other parts)
        own_cnt = np.take_along_axis(nbr_cnt, parts[:, None], axis=1).ravel()
        boundary = np.flatnonzero(self.degrees > own_cnt)
        if len(boundary) == 0:
            return False

        _, _, b_best_gain, _ = boundary_gains(
            nbr_cnt[boundary], parts, boundary
        )

        order = np.argsort(b_best_gain)[::-1]

        moves = []
        cum_gain = best_cum_gain = 0
        best_idx = -1
        max_no_improve = max(int(len(boundary)**0.5), self.nparts)
        no_improve = 0

        for v in iter_struct(boundary[order]):
            src = int(parts[v])
            row = nbr_cnt[v]
            dst, own = self._best_dest(row, src)
            g = int(row[dst] - own)

            if g <= -2:
                continue

            if not can_move(part_wts, src, dst, self.vwts[v], upper, lower):
                continue

            self.move(v, src, dst)
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

        # Rollback moves after the best prefix
        self.rollback(moves, best_idx)

        return best_cum_gain > 0


class TwoWayFMRefiner(BaseRefiner):
    def __init__(self, graph, parts):
        super().__init__(graph, parts, 2)

        nv, vwts = graph.nv, graph.vwts
        self.nc = nc = graph.ncon
        tot = np.maximum(vwts.sum(axis=0), 1)
        self.dominant = (vwts / tot).argmax(axis=1).tolist()

        # Weight of the edges each vertex has inside and outside its side
        idw = np.take_along_axis(self.nbr_cnt, parts[:, None], axis=1)
        idw = idw.ravel().astype(np.int64)
        edw = self.degrees - idw
        self.side, self.gain = parts.tolist(), (edw - idw).tolist()
        self.idw, self.edw = idw.tolist(), edw.tolist()
        self.stamp, self.moved, self.moves = [0]*nv, bytearray(nv), []
        self.cut = int(edw.sum()) // 2

        # Heap the boundary vertices of each side and constraint by gain
        self.heaps = [[[] for _ in range(nc)] for _ in range(2)]
        for v in np.flatnonzero(edw > 0).tolist():
            heap = self.heaps[self.side[v]][self.dominant[v]]
            heap.append((-self.gain[v], 0, v))
        for h in self.heaps[0] + self.heaps[1]:
            heapq.heapify(h)

    def pop(self, rtargets, side=None):
        # Best boundary vertex of the most overweight side and constraint
        order = np.argsort(-self.part_wts / rtargets, axis=None).tolist()
        for k in order:
            src, c = divmod(k, self.nc)
            if side is not None and src != side:
                continue

            heap = self.heaps[src][c]
            while heap:
                _, s, v = heapq.heappop(heap)
                if s == self.stamp[v] and not self.moved[v]:
                    return v

        return -1

    def move(self, v):
        side = self.side
        src = side[v]
        dst = 1 - src
        side[v], self.moved[v] = dst, 1
        self.part_wts[src] -= self.vwts[v]
        self.part_wts[dst] += self.vwts[v]
        self.cut -= self.gain[v]
        self.moves.append(v)

        # Update the gains of the neighbours
        idw, edw, gain = self.idw, self.edw, self.gain
        eidx = slice(self.vtab[v], self.vtab[v + 1])
        for w, ew in zip(self.etab[eidx].tolist(), self.ewts[eidx].tolist()):
            if side[w] == dst:
                idw[w] += ew
                edw[w] -= ew
            else:
                idw[w] -= ew
                edw[w] += ew

            if not self.moved[w]:
                gain[w] = edw[w] - idw[w]
                self.stamp[w] += 1
                if edw[w] > 0:
                    heapq.heappush(self.heaps[side[w]][self.dominant[w]],
                                   (-gain[w], self.stamp[w], w))

    def rollback(self, best_idx):
        for v in self.moves[best_idx + 1:]:
            src = self.side[v]
            self.side[v] = 1 - src
            self.part_wts[src] -= self.vwts[v]
            self.part_wts[1 - src] += self.vwts[v]

        self.parts[:] = self.side

    def refine(self, upper, rtargets, limit):
        part_wts, vwts = self.part_wts, self.vwts

        def excess(wts):
            # Score by the worst then summed squared relative excess
            over = np.maximum((wts - upper) / rtargets, 0)
            return over.max(), (over*over).sum()

        key = excess(part_wts)
        tol, mincut, balancing = key[0], self.cut, key[0] > 0
        best_idx, no_improve = -1, 0

        while no_improve <= limit:
            # Balance by moving vertices off the side with the worst excess
            if balancing:
                over = (part_wts - upper) / rtargets
                side = int(over.max(axis=1).argmax())
            else:
                side = None

            # Take the best vertex of the most overweight side and constraint
            if (v := self.pop(rtargets, side)) < 0:
                break

            # Skip balancing moves which would not reduce the excess
            if balancing:
                src = self.side[v]
                new_wts = part_wts.copy()
                new_wts[src] -= vwts[v]
                new_wts[1 - src] += vwts[v]
                new_key = excess(new_wts)
                if new_key >= key:
                    no_improve += 1
                    continue

            self.move(v)

            # Keep balancing moves, then the best cut within the balance
            if balancing:
                key = new_key
                tol, mincut, balancing = key[0], self.cut, key[0] > 0
                best_idx, no_improve = len(self.moves) - 1, 0
            elif self.cut < mincut and excess(part_wts)[0] <= tol:
                mincut, best_idx, no_improve = self.cut, len(self.moves) - 1, 0
            else:
                no_improve += 1

        # Roll back the moves after the best state
        self.rollback(best_idx)

        return best_idx >= 0
