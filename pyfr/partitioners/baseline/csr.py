import numpy as np


def wbincount(keys, vwts, n):
    return np.column_stack(
        [np.bincount(keys, col, minlength=n) for col in vwts.T]
    )


class CSRGraph:
    def __init__(self, vtab, etab, ewts, vwts, esrc=None):
        self.vtab, self.etab, self.ewts, self.vwts = vtab, etab, ewts, vwts
        self.nv, self.ne, self.ncon = len(vtab) - 1, len(etab), vwts.shape[1]

        # Precompute edge sources
        if esrc is not None:
            self.esrc = esrc
        else:
            vidx = np.arange(self.nv, dtype=etab.dtype)
            self.esrc = np.repeat(vidx, np.diff(vtab))

    def part_wts(self, parts, nparts):
        return wbincount(parts, self.vwts, nparts)

    def cut(self, parts):
        cut = self.ewts[parts[self.esrc] != parts[self.etab]].sum()
        return int(cut) // 2

    def incident(self, vidx):
        # Indices of the edges leaving each vertex in vidx, in vertex order
        vtab = self.vtab
        deg = vtab[vidx + 1] - vtab[vidx]
        off = np.repeat(vtab[vidx] - np.cumsum(deg) + deg, deg)
        return off + np.arange(deg.sum()), deg

    def bfs(self, start):
        levels = np.full(self.nv, -1, dtype=np.int32)
        levels[start] = 0

        frontier = np.array([start], dtype=self.etab.dtype)
        level = 1

        while len(frontier) > 0:
            eidx, _ = self.incident(frontier)

            # Vectorized neighbour gather
            neighbours = self.etab[eidx]

            # Filter to unvisited
            unvisited_mask = levels[neighbours] < 0
            if not unvisited_mask.any():
                break

            unvisited = neighbours[unvisited_mask]
            levels[unvisited] = level

            # Deduplicate frontier via bincount
            seen = np.bincount(unvisited, minlength=self.nv)
            frontier = np.flatnonzero(seen)
            level += 1

        return levels

    def components(self, parts):
        vtab, etab, nv = self.vtab, self.etab, self.nv

        # Select the edges internal to each partition
        same = parts[self.esrc] == parts[etab]
        has_edges = np.diff(vtab) > 0
        starts = vtab[:-1][has_edges]

        # Propagate the smallest label through each connected piece
        lab = np.arange(nv)
        while True:
            # Take the least label of each vertex and its partition neighbours
            vals = np.where(same, lab[etab], nv)
            mlab = lab.copy()
            mlab[has_edges] = np.minimum(lab[has_edges],
                                         np.minimum.reduceat(vals, starts))

            # Hook the label each vertex points at onto this smaller label
            nlab = lab.copy()
            np.minimum.at(nlab, lab, mlab)
            nlab = np.minimum(nlab, mlab)

            # Shorten the label chains by pointer jumping
            while True:
                jlab = nlab[nlab]
                if np.array_equal(jlab, nlab):
                    break
                nlab = jlab

            if np.array_equal(nlab, lab):
                break
            lab = nlab

        # Renumber the components consecutively
        return (np.cumsum(np.bincount(lab, minlength=nv) > 0) - 1)[lab]

    def subgraph(self, vidx):
        idt = self.etab.dtype
        nv_sub = len(vidx)

        # Map global to local indices
        gmap = np.full(self.nv, -1, dtype=idt)
        gmap[vidx] = np.arange(nv_sub, dtype=idt)

        # Get edge ranges
        eidx, counts = self.incident(vidx)

        # Vectorized gather of all neighbours
        v_idx = np.repeat(np.arange(nv_sub), counts)
        all_nb = self.etab[eidx]

        # Filter to internal edges
        local_nb = gmap[all_nb]
        valid = local_nb >= 0

        # Build sub CSR
        valid_counts = np.bincount(v_idx[valid], minlength=nv_sub)

        sub_vtab = np.zeros(nv_sub + 1, dtype=int)
        sub_vtab[1:] = valid_counts.cumsum()
        sub_etab = local_nb[valid]
        sub_ewts = self.ewts[eidx][valid]

        return CSRGraph(sub_vtab, sub_etab, sub_ewts, self.vwts[vidx])

    def contract(self, match):
        idt = self.etab.dtype
        nv_c = match.max() + 1

        # Coarse vertex weights
        c_vwts = wbincount(match, self.vwts, nv_c).astype(np.int32)

        # Map edges to coarse vertices and remove self-loops
        c_src = match[self.esrc]
        c_dst = match[self.etab]
        keep = c_src != c_dst

        if not keep.any():
            return CSRGraph(np.zeros(nv_c + 1, dtype=int),
                            np.array([], dtype=idt),
                            np.array([], dtype=np.int32), c_vwts,
                            np.array([], dtype=idt))

        c_src, c_dst, c_ewts = c_src[keep], c_dst[keep], self.ewts[keep]

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
        c_dst = c_dst.astype(idt)

        # Build coarse CSR
        c_vtab = np.zeros(nv_c + 1, dtype=int)
        c_vtab[1:] = np.bincount(c_src, minlength=nv_c).cumsum()

        c_esrc = np.repeat(np.arange(nv_c, dtype=idt), np.diff(c_vtab))

        return CSRGraph(c_vtab, c_dst, c_ewts, c_vwts, c_esrc)
