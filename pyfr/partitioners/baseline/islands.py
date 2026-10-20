import numpy as np

from pyfr.partitioners.baseline.csr import wbincount


class IslandRepair:
    def __init__(self, graph, parts, comp, targets, upper, maximb):
        self.graph, self.parts, self.maximb = graph, parts, maximb
        self.vtab, self.etab, self.ewts = graph.vtab, graph.etab, graph.ewts
        self.vwts = vwts = graph.vwts
        self.targets, self.upper = targets, upper
        self.nparts = nparts = len(targets)
        self.nmoved = self.nannexed = 0

        # Weight and partition of each component
        self.comp, ncomp = comp, comp.max() + 1
        self.cwts = cwts = wbincount(comp, vwts, ncomp)
        self.cpart = cpart = np.empty(ncomp, dtype=np.int32)
        cpart[comp] = parts

        # Keep the heaviest component in every partition
        order = np.lexsort((-cwts.sum(axis=1), cpart))
        skey = cpart[order]
        self.keep = keep = np.zeros(ncomp, dtype=bool)
        keep[order[np.r_[True, skey[1:] != skey[:-1]]]] = True

        # Group the vertices by component
        self.vorder = np.argsort(comp, kind='stable')
        cptr = np.concatenate(([0], np.bincount(comp, minlength=ncomp)))
        self.cptr = cptr.cumsum()

        self.part_wts = wbincount(parts, vwts, nparts)
        self.settled = np.where(keep[comp], parts, -1)

        # Sum the weight of the islands each partition has still to place
        self.pending = wbincount(cpart[~keep], cwts[~keep], nparts)

    def vertices(self, c):
        return self.vorder[self.cptr[c]:self.cptr[c + 1]]

    def conn(self, vidx):
        # Edge weight from the island into each settled partition
        eidx, _ = self.graph.incident(vidx)
        nbp = self.settled[self.etab[eidx]]
        nbm = nbp >= 0
        return np.bincount(nbp[nbm], self.ewts[eidx][nbm],
                           minlength=self.nparts)

    def dest(self, c, conn, imbmax=None):
        src, cw = self.cpart[c], self.cwts[c]
        cand = np.flatnonzero(conn > 0)

        # Never leave the source too far below its target
        left = self.part_wts[src] - cw
        strip = (left < self.targets[src] / (imbmax or self.maximb)).any()

        # Only consider destinations which stay close to balanced
        new_wts = self.part_wts[cand] - self.pending[cand] + cw
        imb = new_wts / np.maximum(self.targets[cand], 1e-10)
        if imbmax is None:
            fits = (new_wts <= self.upper[cand]).all(axis=1)
        else:
            fits = imb.max(axis=1) <= imbmax

        # Flag islands with no settled neighbours
        if len(cand) == 0:
            return -1
        # Islands which cannot move stay put
        elif strip or not fits.any():
            return src
        else:
            cand, imb = cand[fits], imb[fits]

            # With a single constraint only consider well connected ones
            if self.vwts.shape[1] == 1:
                good = conn[cand] >= 0.5*conn[cand].max()
                cand, imb = cand[good], imb[good]

            # Of these take the best balanced, then the best connected
            best = np.lexsort((-conn[cand], imb.max(axis=1)))[0]
            return cand[best]

    def bridge(self, c, maxhops=16):
        # Shortest chain of outside vertices from the island to its body
        p, settled = self.cpart[c], self.settled
        vtab, etab = self.vtab, self.etab
        frontier = self.vertices(c).tolist()
        prev = dict.fromkeys(frontier, -1)
        for _ in range(maxhops):
            nxt = []
            for x in frontier:
                for w in etab[vtab[x]:vtab[x + 1]].tolist():
                    if w in prev:
                        continue

                    # Walk back along the chain once the body is reached
                    if settled[w] == p:
                        chain = []
                        while prev[x] >= 0:
                            chain.append(x)
                            x = prev[x]

                        return chain if self.chain_fits(chain, p) else []

                    prev[w] = x
                    nxt.append(w)

            frontier = nxt

        return []

    def chain_fits(self, chain, p):
        # Annexing the chain must keep the partition within 20% of its target
        new_wts = self.part_wts[p] - self.pending[p]
        new_wts = new_wts + self.vwts[chain].sum(axis=0)
        imb = new_wts / np.maximum(self.targets[p], 1e-10)
        if imb.max() > 1.2:
            return False

        # And must not leave those it is taken from too far below target
        lost = wbincount(self.parts[chain], self.vwts[chain], self.nparts)
        hit = lost.any(axis=1)
        left = self.part_wts[hit] - lost[hit]
        return not (left < self.targets[hit] / 1.2).any()

    def annex(self, src, chain):
        chain = np.array(chain, dtype=np.int64)
        self.nannexed += 1
        self.part_wts -= wbincount(self.parts[chain], self.vwts[chain],
                                   self.nparts)
        self.part_wts[src] += self.vwts[chain].sum(axis=0)
        self.parts[chain] = src
        self.settled[chain] = src
        self.nmoved += 1

        # Mark the islands the chain passes through as settled
        joined = np.unique(self.comp[chain])
        joined = joined[~self.keep[joined]]
        own = joined[self.cpart[joined] == src]
        self.pending[src] -= self.cwts[own].sum(axis=0)
        self.keep[joined] = True

    def settle(self, c, dst):
        src, vidx = self.cpart[c], self.vertices(c)
        self.pending[src] -= self.cwts[c]
        if dst != src:
            self.parts[vidx] = dst
            self.part_wts[src] -= self.cwts[c]
            self.part_wts[dst] += self.cwts[c]
            self.cpart[c] = dst
            self.nmoved += 1

        self.settled[vidx] = dst

    def place(self, c):
        src = self.cpart[c]
        conn = self.conn(self.vertices(c))

        # Islands with no settled neighbour must wait
        if (dst := self.dest(c, conn)) < 0:
            return False

        # If it cannot move within its bounds relax the balance a little
        if dst == src:
            dst = self.dest(c, conn, min(self.maximb, 1.2))

        # Failing that reconnect the island or relax the balance further
        if dst == src:
            chain = self.bridge(c)
            if chain:
                self.annex(src, chain)
            else:
                dst = self.dest(c, conn, self.maximb)

        self.settle(c, dst)
        return True

    def run(self):
        todo = list(np.flatnonzero(~self.keep))
        while todo:
            remaining = []
            for c in todo:
                if self.keep[c]:
                    continue

                # Islands which could not be placed wait for the next round
                if not self.place(c):
                    remaining.append(c)

            if len(remaining) == len(todo):
                break

            todo = remaining

        return self.nmoved
