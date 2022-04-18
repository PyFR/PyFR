# -*- coding: utf-8 -*-

from ast import literal_eval
from collections import defaultdict
import re

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.shapes import BaseShape
from pyfr.util import match_paired_paren, subclasses, subclass_where


class BaseRegion:
    def interior_eles(self, mesh, rallocs):
        pass

    def surface_faces(self, mesh, rallocs, exclbcs=[]):
        from mpi4py import MPI

        sfaces = set()

        # Begin by assuming all faces of all elements are on the surface
        for etype, eidxs in self.interior_eles(mesh, rallocs).items():
            nfaces = len(subclass_where(BaseShape, name=etype).faces)
            sfaces.update((etype, i, j) for i in eidxs for j in range(nfaces))

        # Eliminate any faces with internal connectivity
        con = mesh[f'con_p{rallocs.prank}'].T
        for l, r in con[['f0', 'f1', 'f2']].astype('U4,i4,i1').tolist():
            if l in sfaces and r in sfaces:
                sfaces.difference_update([l, r])

        # Eliminate faces on specified boundaries
        for b in exclbcs:
            if (f := f'bcon_{b}_p{rallocs.prank}') in mesh:
                bcon = mesh[f][['f0', 'f1', 'f2']].astype('U4,i4,i1')
                sfaces.difference_update(bcon.tolist())

        comm, rank, root = get_comm_rank_root()
        reqs, bufs = [], []

        # Next, consider faces on partition boundaries
        for p in rallocs.prankconn[rallocs.prank]:
            con = mesh[f'con_p{rallocs.prank}p{p}']
            con = con[['f0', 'f1', 'f2']].astype('U4,i4,i1').tolist()

            # See which of these faces are on the surface boundary
            sb = np.array([c in sfaces for c in con])
            rb = np.empty_like(sb)

            # Exchange this information with our neighbour
            reqs.append(comm.Isend(sb, rallocs.pmrankmap[p]))
            reqs.append(comm.Irecv(rb, rallocs.pmrankmap[p]))

            bufs.append((con, sb, rb))

        # Wait for the exchanges to finish
        MPI.Request.Waitall(reqs)

        # Use this data to eliminate any shared faces
        for con, sb, rb in bufs:
            sfaces.difference_update(f for b, f in zip(sb & rb, con) if b)

        # Group the remaining faces by element type
        nsfaces = defaultdict(list)
        for etype, eidx, fidx in sfaces:
            nsfaces[etype].append((eidx, fidx))

        # Sort and return
        return {k: sorted(v) for k, v in nsfaces.items()}


class BoundaryRegion(BaseRegion):
    def __init__(self, bcname, nlayers=1):
        self.bcname = bcname
        self.nlayers = nlayers

    def interior_eles(self, mesh, rallocs):
        from mpi4py import MPI

        bc = f'bcon_{self.bcname}_p{rallocs.prank}'
        eset = defaultdict(list)
        comm, rank, root = get_comm_rank_root()

        # Ensure the boundary exists
        bcranks = comm.gather(bc in mesh, root=root)
        if rank == root and not any(bcranks):
            raise ValueError(f'Boundary {self.bcname} does not exist')

        # Determine which of our elements are directly on the boundary
        if bc in mesh:
            for etype, eidx in mesh[bc][['f0', 'f1']].astype('U4,i4'):
                eset[etype].append(eidx)

        # Handle the case where multiple layers have been requested
        if self.nlayers > 1:
            # Load our internal connectivity array
            con = mesh[f'con_p{rallocs.prank}'].T
            con = con[['f0', 'f1']].astype('U4,i4').tolist()

            # Load our partition boundary connectivity arrays
            pcon = {}
            for p in rallocs.prankconn[rallocs.prank]:
                pc = mesh[f'con_p{rallocs.prank}p{p}']
                pc = pc[['f0', 'f1']].astype('U4,i4').tolist()
                pcon[p] = (pc, *np.empty((2, len(pc)), dtype=bool))

            # Tag all elements in the set as belonging to the first layer
            neset = {(k, j): 0 for k, v in eset.items() for j in v}

            # Iteratively grow out the element set
            for i in range(self.nlayers - 1):
                reqs = []

                # Exchange information about recent updates to our set
                for p, (pc, sb, rb) in pcon.items():
                    sb[:] = [neset.get(c, -1) == i for c in pc]

                    # Start the send/recv requests
                    reqs.append(comm.Isend(sb, rallocs.pmrankmap[p]))
                    reqs.append(comm.Irecv(rb, rallocs.pmrankmap[p]))

                # Grow our element set by considering internal connectivity
                for l, r in con:
                    if neset.get(l, -1) == i and r not in neset:
                        neset[r] = i + 1
                        eset[r[0]].append(r[1])
                    elif neset.get(r, -1) == i and l not in neset:
                        neset[l] = i + 1
                        eset[l[0]].append(l[1])

                # Wait for the exchanges to finish
                MPI.Request.Waitall(reqs)

                # Grow our element set by considering adjacent partitions
                for pc, sb, rb in pcon.values():
                    for l, b in zip(pc, rb):
                        if b and l not in neset:
                            neset[l] = i + 1
                            eset[l[0]].append(l[1])

        return {k: sorted(v) for k, v in eset.items()}


class BaseGeometricRegion(BaseRegion):
    name = None

    def interior_eles(self, mesh, rallocs):
        eset = {}

        for shape in subclasses(BaseShape, just_leaf=True):
            f = f'spt_{shape.name}_p{rallocs.prank}'
            if f not in mesh:
                continue

            inside = self.pts_in_region(mesh[f])
            if np.any(inside):
                eset[shape.name] = np.any(inside, axis=0).nonzero()[0].tolist()

        return eset

    def pts_in_region(self, pts):
        pass


class BoxRegion(BaseGeometricRegion):
    name = 'box'

    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1

    def pts_in_region(self, pts):
        pts = np.moveaxis(pts, -1, 0)

        inside = np.ones(pts.shape[1:], dtype=np.bool)
        for l, p, u in zip(self.x0, pts, self.x1):
            inside &= (l <= p) & (p <= u)

        return inside


class ConicalFrustumRegion(BaseGeometricRegion):
    name = 'conical_frustum'

    def __init__(self, x0, x1, r0, r1):
        self.x0 = x0 = np.array(x0)
        self.x1 = x1 = np.array(x1)
        self.r0 = r0
        self.r1 = r1

        # Heading of the centre line
        self.h = (x1 - x0) / np.linalg.norm(x1 - x0)
        self.h_mag = np.linalg.norm(x1 - x0)

    def pts_in_region(self, pts):
        r0, r1 = self.r0, self.r1

        # Project the points onto the centre line
        dotp = (pts - self.x0) @ self.h

        # Compute the distances between the points and their projections
        d = np.linalg.norm(self.x0 + dotp[..., None]*self.h - pts, axis=-1)

        # Interpolate the radii along the centre line
        r = np.interp(dotp, [0, self.h_mag], [r0, r1], left=-1, right=-1)

        # With this determine which points are inside our conical frustum
        return d <= r


class ConeRegion(ConicalFrustumRegion):
    name = 'cone'

    def __init__(self, x0, x1, r):
        super().__init__(x0, x1, r, 0)


class CylinderRegion(ConicalFrustumRegion):
    name = 'cylinder'

    def __init__(self, x0, x1, r):
        super().__init__(x0, x1, r, r)


class EllipsoidRegion(BaseGeometricRegion):
    name = 'ellipsoid'

    def __init__(self, x0, a, b, c):
        self.x0 = np.array(x0)
        self.abc = np.array([a, b, c])

    def pts_in_region(self, pts):
        return np.sum(((pts - self.x0) / self.abc)**2, axis=-1) <= 1


class SphereRegion(EllipsoidRegion):
    name = 'sphere'

    def __init__(self, x0, r):
        super().__init__(x0, r, r, r)


class ConstructiveRegion(BaseGeometricRegion):
    def __init__(self, expr):
        rexprs = []

        # Factor out the individual region expressions
        expr = re.sub(
            r'(\w+)\((' + match_paired_paren('()') + r')\)',
            lambda m: rexprs.append(m.groups()) or f'r{len(rexprs) - 1}',
            expr
        )

        # Parse these region expressions
        self.regions = regions = []
        for name, args in rexprs:
            cls = subclass_where(BaseGeometricRegion, name=name)
            regions.append(cls(*literal_eval(args)))

        # Rewrite in terms of boolean operators
        self.expr = expr.replace('+', '|').replace('-', '&~')

        # Validate
        if not re.match(r'[r0-9|&~() ]+$', self.expr):
            raise ValueError('Invalid region expression')

    def pts_in_region(self, pts):
        # Query each of our constituent regions
        rvars = {f'r{i}': r.pts_in_region(pts)
                 for i, r in enumerate(self.regions)}

        return eval(self.expr, {'__builtins__': None}, rvars)
