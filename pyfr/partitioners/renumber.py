import re

import numpy as np

from pyfr.nputil import morton_keys
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


def face_codec(codec):
    # Element type and face number associated with each codec entry
    ctype, cface = ['']*len(codec), np.full(len(codec), -1)
    for i, c in enumerate(codec):
        if m := re.match(r'eles/(\w+)/face/(\d+)$', c):
            ctype[i], cface[i] = m[1], int(m[2])

    return np.array(ctype), cface


def opposite_faces(shape):
    # Pair each face with the face opposite it, if any
    norms = np.array([n for *_, n in shape.faces])
    anti = (norms == -norms[:, None]).all(axis=2)

    return np.where(anti.any(axis=1), np.argmax(anti, axis=1), -1)


def chain_faces(shape):
    opp = opposite_faces(shape)

    # Take the faces which lack opposites and share their kind with another
    fk = [k if o < 0 else None for (k, *_), o in zip(shape.faces, opp)]
    return [i for i, k in enumerate(fk) if k and fk.count(k) > 1]


class ElementRenumberer:
    def __init__(self, codec, grpsz=32, linesz=16, blksz=256, margin=0.05):
        # Number of interface points whose partner data is fetched together
        self.grpsz = grpsz

        # Number of consecutive elements which share a memory line
        self.linesz = linesz

        # Number of consecutive elements the interface views visit together
        self.blksz = blksz

        # Fraction by which a new ordering must beat the native ordering
        self.margin = margin

        self._ctype, self._cface = face_codec(codec)

    @staticmethod
    def _new_idx(order):
        idx = np.empty(len(order), dtype=int)
        idx[order] = np.arange(len(order))

        return idx

    def _morton_order(self, pts):
        return self._new_idx(np.argsort(morton_keys(pts)))

    @staticmethod
    def _face_pairs(opp):
        # Take the lower numbered face of each opposite pair
        fa = np.flatnonzero(opp > np.arange(len(opp)))

        return np.stack((fa, opp[fa]), axis=1)

    @staticmethod
    def _follow_faces(nxt):
        s = np.arange(len(nxt))

        # Label each face with the smallest face its chord reaches
        lo, b = s.copy(), nxt
        while True:
            old, lo = lo, np.minimum(lo, lo[b])
            if np.array_equal(lo, old):
                break

            b = b[b]

        # Cut each cycle at its smallest face
        cut = lo[nxt] == s
        nxt[cut] = s[cut]

        # Find the end of each chord and the distance of each face from it
        dists, ends = (nxt != s).astype(int), nxt
        while True:
            old, ends = ends, ends[ends]
            if np.array_equal(ends, old):
                break

            dists += dists[old]

        return ends, dists

    def _chord_trace(self, nbr, nbf, opp):
        n, nf = nbr.shape
        fa, fb = self._face_pairs(opp).T

        # Leave each element through a face and carry on through its opposite
        ok = (nbr >= 0) & (opp >= 0) & (opp[nbf] >= 0)
        e, g = np.nonzero(ok)
        nxt = np.arange(n*nf)
        nxt[e*nf + g] = nbr[e, g]*nf + opp[nbf[e, g]]
        ends, dists = [a.reshape(n, nf) for a in self._follow_faces(nxt)]

        # Identify each chord by its ends and position elements along it
        ea, eb = ends[:, fa], ends[:, fb]
        cids = np.minimum(ea, eb)
        pos = np.where(ea < eb, dists[:, fa], dists[:, fb])

        # Flag the chords which close on themselves
        cyc = ok.flat[ea]

        return cids, pos, cyc

    def _chord_select(self, nbr, nbf, opp, longest):
        # Follow the chords through each element
        cids, pos, cyc = self._chord_trace(nbr, nbf, opp)

        # Take the longest or shortest chord through each element
        clens = np.bincount(cids.ravel())
        e = np.arange(len(cids))
        p = (np.argmax if longest else np.argmin)(clens[cids], axis=1)
        cs, ps, L = cids[e, p], pos[e, p], clens[cids[e, p]]
        cs = np.unique(cs, return_inverse=True)[1]

        return cs, ps, L, p, cyc[e, p]

    def _chord_order(self, cen, nbr, nbf, opp, longest):
        # Select the chord to follow through each element
        cs, ps, L, p, cyc = self._chord_select(nbr, nbf, opp, longest)

        # Chain the chords themselves through their transverse neighbours
        if lvl := self._chord_conn(cs, ps, L, p, cyc, nbr, opp):
            (tc, tf, topp), shifts = lvl
            cidxs = self._chord_order(self._centroids(cen, cs), tc, tf, topp,
                                      longest)
            ps = self._chord_align(cidxs, shifts, cs, ps, L, cyc)
        else:
            # Otherwise order line-long chord segments along a Morton curve
            nseg = L.max() // self.linesz + 1
            cs = np.unique(cs*nseg + ps // self.linesz, return_inverse=True)[1]
            cidxs = self._morton_order(self._centroids(cen, cs))

        return self._new_idx(np.lexsort((ps, cidxs[cs])))

    @staticmethod
    def _centroids(cen, groups):
        # Average the centroids of the elements in each group
        cnt = np.bincount(groups)

        return np.stack([np.bincount(groups, c) / cnt for c in cen.T], axis=1)

    @staticmethod
    def _chord_align(cidxs, shifts, cs, ps, L, cyc):
        (keys, us, uc), nc = shifts, len(cidxs)

        # Bring consecutive aligned chords into a common frame
        co = np.argsort(cidxs)
        pk = co[:-1]*nc + co[1:]
        ix = np.minimum(np.searchsorted(keys, pk), len(keys) - 1)
        hit = keys[ix] == pk
        S, C = np.ones(nc, dtype=int), np.zeros(nc, dtype=int)
        S[co[1:]] = np.cumprod(np.where(hit, us[ix], 1))
        C[co[1:]] = np.cumsum(np.where(hit, uc[ix], 0)*S[co[:-1]])

        # Wrap the positions of chords which close on themselves
        ps = S[cs]*ps + C[cs]

        return np.where(cyc, ps % L, ps)

    @staticmethod
    def _group_value(groups, v, ngroups):
        # Find the value shared by every member of each group, if any
        lo = np.full(ngroups, v.max(initial=0))
        hi = np.full(ngroups, v.min(initial=0))
        np.minimum.at(lo, groups, v)
        np.maximum.at(hi, groups, v)

        return lo == hi, lo

    def _chord_shifts(self, cs, ps, L, pf, cyc, nbr, nc):
        nf = nbr.shape[1]

        # Find the neighbours through faces our chord does not run along
        along = (np.arange(nf) == pf[..., None]).any(axis=1)
        e, g = np.nonzero((nbr >= 0) & ~along)
        ee, Le, ce = nbr[e, g], L[e], cyc[e]
        keys, inv = np.unique(cs[e]*nc + cs[ee], return_inverse=True)
        nk = len(keys)

        # Check every element maps onto the neighbouring chord the same way
        ds, dr = ps[e] - ps[ee], ps[e] + ps[ee]
        oks, ucs = self._group_value(inv, np.where(ce, ds % Le, ds), nk)
        okr, ucr = self._group_value(inv, np.where(ce, dr % Le, dr), nk)
        ok = oks | okr

        # Shift positions on the neighbouring chord onto our own
        us, uc = np.where(oks, 1, -1), np.where(oks, ucs, ucr)

        return keys[ok], us[ok], uc[ok]

    def _chord_conn(self, cs, ps, L, p, cyc, nbr, opp):
        nc = cs.max() + 1

        # Chords can only be chained through a second pair of faces
        sf = self._face_pairs(opp)
        if len(sf) < 2:
            return None

        # Compute the position shifts between aligned neighbouring chords
        keys, us, uc = self._chord_shifts(cs, ps, L, sf[p], cyc, nbr, nc)
        if not len(keys):
            return None

        # Take the face pairs a chord does not run along as its slots
        o = np.lexsort((ps, cs))[::-1]
        r0 = np.empty(nc, dtype=int)
        r0[cs[o]] = o
        oth = np.arange(len(sf)) != p[r0, None]
        slots = np.broadcast_to(sf, (nc, *sf.shape))[oth].reshape(nc, -1)

        # Find the neighbouring chord through each slot, if aligned with us
        tn = np.take_along_axis(nbr[r0], slots, 1)
        tc = np.where(tn >= 0, cs[tn], -1)
        tc[~np.isin(np.arange(nc)[:, None]*nc + tc, keys)] = -1

        # Find the slot of the neighbouring chord through which we arrive
        back = tc[tc] == np.arange(nc)[:, None, None]
        tc[~back.any(axis=2)] = -1
        topp = np.arange(slots.shape[1]) ^ 1

        return (tc, np.argmax(back, axis=2), topp), (keys, us, uc)

    def _partner_lines(self, lhs, rhs, nf, idx):
        (leles, lfaces), (reles, rfaces) = lhs, rhs
        n, blksz = len(idx), self.blksz
        linesz, grpsz = self.linesz, self.grpsz

        # Key each side by (face, position) with the smaller key as the LHS
        lk, rk = lfaces*n + idx[leles], rfaces*n + idx[reles]
        lo, hi = np.minimum(lk, rk), np.maximum(lk, rk)

        # Compute the partner memory line of each LHS point
        hf, hp = divmod(hi, n)
        lines = hf*(n // linesz + 1) + hp // linesz

        # Order the LHS points as the interface view does: by block then face
        vf, vp = divmod(lo, n)
        vb, vo = divmod(vp, blksz)
        vkey = (vb*nf + vf)*blksz + vo
        rl = np.full((n // blksz + 1)*nf*blksz, -1)
        rl[vkey] = lines
        rl = rl[rl >= 0]

        # Count the distinct partner lines touched by each group of points
        rl = np.pad(rl, (0, -len(rl) % grpsz), mode='edge').reshape(-1, grpsz)
        nl = np.count_nonzero(np.diff(np.sort(rl), axis=1), axis=1) + 1

        return nl.sum() / max(len(nl), 1)

    def _renumber_etype(self, etype, cen, nbr, nbf):
        n, nf = nbr.shape

        # Deduplicate the interfaces between elements of this type
        fkey = np.arange(n*nf).reshape(n, nf)
        lhs = np.nonzero((nbr >= 0) & (fkey < nbr*nf + nbf))
        rhs = nbr[lhs], nbf[lhs]

        # Start with the native and space-filling curve orderings
        cands = [np.arange(n), self._morton_order(cen)]

        # Identify tensor-product directions, if any, for this element type
        shape = subclass_where(BaseShape, name=etype)
        opp = opposite_faces(shape)

        # Reoriented meshes also run chains through their first chain faces
        if cf := chain_faces(shape):
            opp[cf[0]], opp[cf[1]] = cf[1], cf[0]

        # If the element has opposing faces then number along the grid lines
        if len(pairs := self._face_pairs(opp)):
            # With several pairs try favouring the short and long lines
            for longest in (False, True)[:len(pairs)]:
                cands.append(self._chord_order(cen, nbr, nbf, opp, longest))

        # Score each ordering by how many lines its partner accesses touch
        scores = [self._partner_lines(lhs, rhs, nf, c) for c in cands]

        # Keep the native ordering unless a candidate is clearly better
        scores[0] *= 1 - self.margin

        return cands[np.argmin(scores)]

    def _periodic_faces(self, eles, periodic):
        pmask = {et: np.zeros(einfo['faces'].shape, dtype=bool)
                 for et, einfo in eles.items()}

        # Flag the element faces which are paired across a periodic boundary
        for pcon in periodic:
            pcon = pcon.reshape(-1)
            for et, fm in pmask.items():
                m = self._ctype[pcon['cidx']] == et
                fm[pcon['off'][m], self._cface[pcon['cidx'][m]]] = True

        return pmask

    def renumber(self, nodes, eles, eclass, periodic=()):
        eidx = {}
        pmask = self._periodic_faces(eles, periodic)
        for etype, einfo in eles.items():
            # Compute the element centroids and bounding boxes
            epts = nodes[einfo['nodes'].T]
            cen, ewid = epts.mean(axis=0), np.ptp(epts, axis=0)

            # Measure the centroids in typical element widths along each axis
            cen /= np.median(ewid, axis=0)

            # Find the neighbours of the same type and the faces we enter by
            offs, cidxs = einfo['faces']['off'], einfo['faces']['cidx']
            same = (self._ctype[cidxs] == etype) & ~pmask[etype]
            nbr = np.where(same, offs, -1)
            nbf = np.where(same, self._cface[cidxs], -1)

            # Number the elements of each class as a separate run
            eidx[etype] = idx = np.empty(len(einfo), dtype=int)
            for c in np.unique(eclass[etype]):
                sel = np.flatnonzero(eclass[etype] == c)

                # Keep only the neighbours which are in the same class
                loc = np.full(len(einfo), -1)
                loc[sel] = np.arange(len(sel))
                cnbr = np.where(nbr[sel] >= 0, loc[nbr[sel]], -1)
                cnbf = np.where(cnbr >= 0, nbf[sel], -1)

                idx[sel] = self._renumber_etype(etype, cen[sel], cnbr, cnbf)

        return eidx
