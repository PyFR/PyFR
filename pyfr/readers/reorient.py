import itertools as it

import numpy as np

from pyfr.nputil import iter_struct, morton_keys
from pyfr.partitioners.renumber import chain_faces, face_codec
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


class ElementReorienter:
    def __init__(self, codec):
        self.codec = codec
        self._ctype, self._cface = face_codec(codec)

    @staticmethod
    def _match(pts, ipts):
        # Find the point in pts which coincides with each point in ipts
        d = np.linalg.norm(pts[:, None] - ipts[None], axis=2)

        return np.argmin(d, axis=0)

    def _rotations(self, shape, nspts):
        verts = shape.std_ele(1)
        spts = shape.std_ele(shape.order_from_npts(nspts))

        # Find the centroid of each face in reference coordinates
        fcen = []
        for kind, proj, _ in shape.faces:
            fverts = subclass_where(BaseShape, name=kind).std_ele(1)
            fcen.append(proj(*fverts.mean(axis=0)))
        fcen = np.array(fcen)

        V = np.column_stack((verts, np.ones(len(verts))))
        for p in it.permutations(range(len(verts))):
            # Fit the affine map which takes each vertex onto its image
            img = verts[list(p)]
            M = np.linalg.lstsq(V, img, rcond=None)[0]
            A, b = M[:-1], M[-1]

            # Keep the maps which are rotations of the reference element
            if np.allclose(V @ M, img) and np.linalg.det(A) > 0:
                yield (self._match(fcen, fcen @ A + b),
                       self._match(spts @ A + b, spts))

    def _chain_rotations(self, shape, nspts, cf):
        nf = len(shape.faces)

        # Default to leaving the element as it is
        fperm = np.tile(np.arange(nf), (nf, nf, 1))
        sperm = np.tile(np.arange(nspts), (nf, nf, 1))
        adm = np.zeros((nf, nf), dtype=bool)

        # Index each rotation by the faces it sends to the chain faces
        for fp, sp in self._rotations(shape, nspts):
            a, b = np.argsort(fp)[cf[:2]]
            fperm[a, b], sperm[a, b], adm[a, b] = fp, sp, True

        return fperm, sperm, adm

    @staticmethod
    def _grow_chains(nbr, nbf, adm, mkeys):
        n = len(nbr)
        fin, fout, seen = [-1]*n, [-1]*n, [False]*n
        nbrl, nbfl, adml, mkeysl = (a.tolist() for a in (nbr, nbf, adm, mkeys))

        # Grow a chain from each unvisited element in turn
        for s in iter_struct(np.argsort(mkeys)):
            if seen[s]:
                continue

            e, f = s, -1
            while True:
                seen[e] = True

                # Move to the free neighbour with the fewest free neighbours
                best = None
                for g, p in enumerate(nbrl[e]):
                    if p >= 0 and not seen[p] and (f < 0 or adml[f][g]):
                        free = sum(not seen[q] for q in nbrl[p] if q >= 0)
                        if best is None or (free, mkeysl[p]) < best[0]:
                            best = (free, mkeysl[p]), g, p

                if best is None:
                    break

                # Record the exit and entry faces and step along the chain
                _, g, p = best
                fout[e], fin[p] = g, nbfl[e][g]
                e, f = p, nbfl[e][g]

        return np.array(fin), np.array(fout)

    def _remap_faces(self, faces, fcmaps):
        cidxs, offs = faces['cidx'], faces['off']

        # Update the face numbers of references to reoriented elements
        for etype, fcmap in fcmaps.items():
            m = self._ctype[cidxs] == etype
            cidxs[m] = fcmap[offs[m], self._cface[cidxs[m]]]

    def _reorient_etype(self, nodes, shape, cf, einfo):
        nspts = einfo['nodes'].shape[1]
        fperm, sperm, adm = self._chain_rotations(shape, nspts, cf)

        # Find the neighbours of the same type through the chain faces
        offs, cidxs = einfo['faces']['off'], einfo['faces']['cidx']
        same = self._ctype[cidxs] == shape.name
        same &= np.isin(np.arange(len(shape.faces)), cf)
        nbr = np.where(same, offs, -1)
        nbf = np.where(same, self._cface[cidxs], -1)

        # Order the elements along a Morton curve of their centroids
        mkeys = morton_keys(nodes[einfo['nodes']].mean(axis=1))

        # Cover the elements with chains
        fin, fout = self._grow_chains(nbr, nbf, adm, mkeys)

        # Give the ends of chains any admissible second face
        a = np.where(fin >= 0, fin, np.argmax(adm[:, fout], axis=0))
        b = np.where(fout >= 0, fout, np.argmax(adm[a], axis=1))

        # Leave elements which are not on a chain as they are
        lone = (fin < 0) & (fout < 0)
        a[lone] = b[lone] = 0

        # Rotate each element so its chain enters and exits canonically
        einfo['nodes'] = np.take_along_axis(einfo['nodes'], sperm[a, b],
                                            axis=1)

        return fperm[a, b]

    def reorient(self, nodes, eles, periodic):
        fperms, fcmaps = {}, {}
        for etype, einfo in eles.items():
            shape = subclass_where(BaseShape, name=etype)

            # Skip element types which lack chain faces
            if not (cf := chain_faces(shape)):
                continue

            fperms[etype] = fp = self._reorient_etype(nodes, shape, cf, einfo)

            # Map each old face onto the codec entry of its new face
            fnames = [f'eles/{etype}/face/{i}' for i in range(fp.shape[1])]
            fcidxs = np.array([self.codec.index(f) for f in fnames])
            fcmaps[etype] = fcidxs[fp]

        # Update the face numbers of neighbouring and periodic faces
        for einfo in eles.values():
            self._remap_faces(einfo['faces'], fcmaps)
        for pidx, *_ in periodic.values():
            self._remap_faces(pidx, fcmaps)

        # Renumber the faces of the reoriented elements themselves
        for etype, fp in fperms.items():
            faces = eles[etype]['faces']
            np.put_along_axis(faces, fp, faces.copy(), axis=1)
