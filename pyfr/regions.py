from ast import literal_eval
from collections import defaultdict
import re

import numpy as np
from rtree.index import Index, Property

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.shapes import BaseShape
from pyfr.util import match_paired_paren, subclass_where


def parse_region_expr(expr, rdata=None):
    # Geometric region
    if '(' in expr:
        return ConstructiveRegion(expr, rdata)
    # Boundary region
    else:
        return BoundaryRegion(expr)


class BaseRegion:
    def interior_eles(self, mesh):
        pass

    def surface_faces(self, mesh, exclbcs=[]):
        comm, rank, root = get_comm_rank_root()
        sfaces = set()

        # Begin by assuming all faces of all elements are on the surface
        for etype, eidxs in self.interior_eles(mesh).items():
            nfaces = len(subclass_where(BaseShape, name=etype).faces)
            sfaces.update((etype, i, j) for i in eidxs for j in range(nfaces))

        # Eliminate any faces with internal connectivity
        for l, r in zip(*mesh.con):
            if l in sfaces and r in sfaces:
                sfaces.difference_update([l, r])

        # Eliminate faces on specified boundaries
        for b in exclbcs:
            sfaces.difference_update(mesh.bcon.get(b, set()))

        reqs, bufs = [], []

        # Next, consider faces on partition boundaries
        for p, con in mesh.con_p.items():
            # See which of these faces are on the surface boundary
            sb = np.array([c in sfaces for c in con])
            rb = np.empty_like(sb)

            # Exchange this information with our neighbour
            reqs.append(comm.Isendrecv(sb, p, recvbuf=rb, source=p))
            bufs.append((con, sb, rb))

        # Wait for the exchanges to finish
        mpi.Request.Waitall(reqs)

        # Use this data to eliminate any shared faces
        for con, sb, rb in bufs:
            sfaces.difference_update(f for b, f in zip(sb & rb, con) if b)

        # Group the remaining faces by element type
        nsfaces = defaultdict(list)
        for etype, eidx, fidx in sfaces:
            nsfaces[etype, fidx].append(eidx)

        # Sort and return
        return {k: sorted(v) for k, v in nsfaces.items()}

    @staticmethod
    def expand(mesh, eles, nlayers):
        comm, rank, root = get_comm_rank_root()
        eles = defaultdict(list, eles)

        # Load our internal connectivity array
        con = [(l[:2], r[:2]) for l, r in zip(*mesh.con)]

        # Load our partition boundary connectivity arrays
        pcon = {}
        for p, pc in mesh.con_p.items():
            pc = [(etype, eidx) for etype, eidx, fidx in pc]
            pcon[p] = (pc, np.empty(len(pc), dtype=bool))

        # Tag all elements in the set as belonging to the first layer
        neles = {(k, j): 0 for k, v in eles.items() for j in v}

        # Iteratively grow out the element set
        for i in range(nlayers):
            reqs = []

            # Exchange information about recent updates to our set
            for p, (pc, sb) in pcon.items():
                sb[:] = [neles.get(c, -1) == i for c in pc]

                # Start the send/recv requests
                reqs.append(comm.Isendrecv_replace(sb, dest=p, source=p))

            # Grow our element set by considering internal connectivity
            for l, r in con:
                if neles.get(l, -1) == i and r not in neles:
                    neles[r] = i + 1
                    eles[r[0]].append(r[1])
                elif neles.get(r, -1) == i and l not in neles:
                    neles[l] = i + 1
                    eles[l[0]].append(l[1])

            # Wait for the exchanges to finish
            mpi.Request.Waitall(reqs)

            # Grow our element set by considering adjacent partitions
            for pc, rb in pcon.values():
                for l, b in zip(pc, rb):
                    if b and l not in neles:
                        neles[l] = i + 1
                        eles[l[0]].append(l[1])

        return eles


class BoundaryRegion(BaseRegion):
    def __init__(self, bcname):
        self.bcname = bcname

    def interior_eles(self, mesh):
        comm, rank, root = get_comm_rank_root()

        eset = defaultdict(list)

        # Ensure the boundary exists
        bcranks = comm.gather(self.bcname in mesh.bcon, root=root)
        if rank == root and not any(bcranks):
            raise ValueError(f'Boundary {self.bcname} does not exist')

        # Determine which of our elements are directly on the boundary
        if self.bcname in mesh.bcon:
            for etype, eidx, fidx in mesh.bcon[self.bcname]:
                eset[etype].append(eidx)

        return {k: sorted(v) for k, v in eset.items()}


class BaseGeometricRegion(BaseRegion):
    name = None

    def __init__(self, rdata=None, rot=None):
        match rot:
            case None:
                self.rot = None
            case [phi, theta, psi]:
                phi, theta, psi = np.deg2rad([phi, theta, psi])

                c, s = np.cos(phi), np.sin(phi)
                Z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                c, s = np.cos(theta), np.sin(theta)
                Y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

                c, s = np.cos(psi), np.sin(psi)
                X = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

                self.rot = Z @ Y @ X
            case theta:
                theta = np.deg2rad(theta)

                c, s = np.cos(theta), np.sin(theta)
                self.rot = np.array([[c, -s], [s, c]])

    def interior_eles(self, mesh):
        eset = {}

        for etype, spts in mesh.spts.items():
            inside = self.pts_in_region(np.mean(spts, axis=0))
            inside[~inside] = self.pts_in_region(spts[:, ~inside]).any(axis=0)

            eset[etype] = inside.nonzero()[0].tolist()

        return {k: sorted(v) for k, v in eset.items()}

    def pts_in_region(self, pts):
        if self.rot is not None:
            pts = np.einsum('ij,klj->kli', self.rot.T, pts)

        return self._pts_in_region(pts)


class BoxRegion(BaseGeometricRegion):
    name = 'box'

    def __init__(self, x0, x1, **kwargs):
        super().__init__(**kwargs)

        self.x0 = x0
        self.x1 = x1

    def _pts_in_region(self, pts):
        pts = np.moveaxis(pts, -1, 0)

        inside = np.ones(pts.shape[1:], dtype=bool)
        for l, p, u in zip(self.x0, pts, self.x1):
            inside &= (l <= p) & (p <= u)

        return inside


class ConicalFrustumRegion(BaseGeometricRegion):
    name = 'conical_frustum'

    def __init__(self, x0, x1, r0, r1, **kwargs):
        super().__init__(**kwargs)

        self.x0 = x0 = np.array(x0)
        self.x1 = x1 = np.array(x1)
        self.r0 = r0
        self.r1 = r1

        # Heading of the centre line
        self.h = (x1 - x0) / np.linalg.norm(x1 - x0)
        self.h_mag = np.linalg.norm(x1 - x0)

    def _pts_in_region(self, pts):
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

    def __init__(self, x0, x1, r, **kwargs):
        super().__init__(x0, x1, r, 0, **kwargs)


class CylinderRegion(ConicalFrustumRegion):
    name = 'cylinder'

    def __init__(self, x0, x1, r, **kwargs):
        super().__init__(x0, x1, r, r, **kwargs)


class EllipsoidRegion(BaseGeometricRegion):
    name = 'ellipsoid'

    def __init__(self, x0, a, b, c, **kwargs):
        super().__init__(**kwargs)

        self.x0 = np.array(x0)
        self.abc = np.array([a, b, c])

    def _pts_in_region(self, pts):
        return np.sum(((pts - self.x0) / self.abc)**2, axis=-1) <= 1


class SphereRegion(EllipsoidRegion):
    name = 'sphere'

    def __init__(self, x0, r, **kwargs):
        super().__init__(x0, r, r, r, **kwargs)


class STLRegion(BaseGeometricRegion):
    name = 'stl'

    def __init__(self, name, rdata, **kwargs):
        super().__init__(rdata=rdata, **kwargs)

        stl = rdata[f'stl/{name}']
        if not stl.attrs['closed']:
            raise ValueError('STL region must be closed')

        faces = stl[:, 1:].astype(float)

        # Compute the global bounding box
        self.x0 = faces.min(axis=(0, 1))
        self.x1 = faces.max(axis=(0, 1))

        # Compute the face direction vectors
        fa, fb, fc = faces.swapaxes(0, 1)
        e1, e2 = fb - fa, fc - fa

        # Compute the associated normals
        norms = np.cross(e1, e2)

        # Prune faces which are parallel to the z-axis
        valid = np.abs(norms[:, 2]) > 1e-6
        faces, norms = faces[valid], norms[valid]
        fa, e1, e2 = fa[valid], e1[valid], e2[valid]

        # Compute the Barycentric and intersection point projection matrix
        z = np.zeros(len(fa))
        mat = (-1 / norms[:, 2])*np.array([
            [-e2[:, 1], e2[:, 0], z],
            [e1[:, 1], -e1[:, 0], z],
            norms.T
        ])

        # Save the required face data
        self.fa = fa.copy()
        self.mat = np.moveaxis(mat, 2, 0).copy()

        # Compute the per-face bounding box
        tbox = np.hstack([faces.min(axis=1) - 1e-6,
                            faces.max(axis=1) + 1e-6])

        # Use this to construct an R-tree index
        ins = ((i, p.tolist(), None) for i, p in enumerate(tbox))
        props = Property(dimension=3, interleaved=True)
        self.tri_idx = Index(ins, properties=props)

    def _pts_in_region(self, pts):
        inside = np.ones(pts.shape[:-1], dtype=bool)
        finside = inside.reshape(-1)

        # Exclude points outside the bounding box
        for i, (l, u) in enumerate(zip(self.x0, self.x1)):
            inside &= (l <= pts[..., i]) & (pts[..., i] <= u)

        # Iterate through the remaining points
        cidx = np.nonzero(finside)[0]
        for i, ro in zip(cidx, pts.reshape(-1, 3)[cidx]):
            crossings = 0

            # Count how many times a ray cast in +z from this point
            # itersects a face on our surface
            tbox = [*ro, *ro[:2], self.x1[2]]
            for j in self.tri_idx.intersection(tbox, objects=False):
                u, v, t = np.dot(self.mat[j], (ro - self.fa[j]).T).tolist()
                crossings += t >= 0 and u >= 0 and v >= 0 and (u + v) <= 1

            finside[i] = (crossings % 2) == 1

        return inside


class ConstructiveRegion(BaseGeometricRegion):
    def __init__(self, expr, rdata=None):
        super().__init__()

        # Factor out the individual region expressions
        rexprs = []
        self.expr = re.sub(
            r'(\w+)\((' + match_paired_paren('()') + r')\)',
            lambda m: rexprs.append(m.groups()) or f'r{len(rexprs) - 1}',
            expr
        )

        # Validate
        if not re.match(r'[r0-9() +-]+$', self.expr):
            raise ValueError('Invalid region expression')

        # Parse these region expressions
        self.regions = regions = []
        for name, args in rexprs:
            cls = subclass_where(BaseGeometricRegion, name=name)

            # Process keyword-only arguments
            for k in ['rot']:
                args = re.sub(fr'{k}\s*=\s*', f'(None, "{k}"), ', args)

            # Evaluate the arguments
            argit = iter(literal_eval(args + ','))

            # Prepare the argument list
            kargs, kwargs = [], {'rdata': rdata}
            for arg in argit:
                match arg:
                    case (None, str()):
                        kwargs[arg[1]] = next(argit)
                    case _:
                        kargs.append(arg)

            # Construct the region
            regions.append(cls(*kargs, **kwargs))

    def pts_in_region(self, pts):
        # Helper to translate + and - to their boolean algebra equivalents
        class RegionVar:
            def __init__(self, r):
                self.r = r

            def __add__(self, rhs):
                return RegionVar(self.r | rhs.r)

            def __sub__(self, rhs):
                return RegionVar(self.r & ~rhs.r)

        # Query each of our constituent regions
        rvars = {f'r{i}': RegionVar(r.pts_in_region(pts))
                 for i, r in enumerate(self.regions)}

        return eval(self.expr, {'__builtins__': None}, rvars).r
