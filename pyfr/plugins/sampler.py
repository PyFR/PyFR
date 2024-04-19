from argparse import FileType
from collections import defaultdict
from pathlib import Path
import re

import h5py
import numpy as np
from rtree.index import Index, Property

from pyfr.mpiutil import get_comm_rank_root, get_start_end_csize, init_mpi, mpi
from pyfr.plugins.base import (BaseCLIPlugin, BaseSolnPlugin, cli_external,
                               init_csv)
from pyfr.polys import get_polybasis
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.util import memoize, subclass_where


class _PointLocator:
    def __init__(self, mesh, fine_order=6):
        self.mesh = mesh
        self.fine_order = fine_order

    def locate(self, pts):
        comm, rank, root = get_comm_rank_root()

        # Allocate the location buffer
        dtype = [('dist', float), ('cidx', np.int16), ('eidx', np.int64),
                 ('tloc', float, self.mesh.ndims)]
        locs = np.zeros(len(pts), dtype=dtype)
        locs['dist'] = np.inf

        # Find the nearest node to each query point
        nearest = self._find_closest_node(pts)

        # Reduce over each of our element types
        for etype, eidxs in self.mesh.eidxs.items():
            cidx = self.mesh.codec.index(f'eles/{etype}')

            eclosest = self._find_closest_element(etype, pts, nearest)
            for i, (dist, eidx, tloc) in eclosest.items():
                l = locs[i]

                if dist < l['dist']:
                    l['dist'], l['tloc'] = dist, tloc
                    l['cidx'], l['eidx'] = cidx, eidxs[eidx]

        # Reduce
        self._minloc(comm.Allreduce, mpi.IN_PLACE, locs, ndim=3)

        # Validate
        if rank == root:
            for i, l in enumerate(locs):
                if l['dist'] == np.inf:
                    ploc = ', '.join(str(p) for p in pts[i])
                    raise ValueError(f'Unable to locate point ({ploc})')

        return locs

    @memoize
    def _get_shape_basis_order(self, etype, nspts):
        shape = subclass_where(BaseShape, name=etype)
        order = shape.order_from_nspts(nspts)
        basis = get_polybasis(etype, order, shape.std_ele(order - 1))

        return shape, basis, order

    def _minloc(self, coll, x, y, ndim=None):
        dtype = y.dtype
        fields = list(dtype.fields)
        fields = fields if ndim is None else fields[:ndim]

        def op(pmem, qmem, dt):
            p = np.frombuffer(pmem, dtype=dtype)
            q = np.frombuffer(qmem, dtype=dtype)

            mask = p[fields[0]] < q[fields[0]]
            for i, f in enumerate(fields[1:], start=1):
                fmask = p[f] < q[f]
                for g in fields[:i]:
                    fmask &= p[g] == q[g]

                mask |= fmask

            q[mask] = p[mask]

        sbuf = (x, mpi.BYTE) if x is not mpi.IN_PLACE else x
        rbuf = (y, mpi.BYTE)
        op = mpi.Op.Create(op, commute=False)

        try:
            coll(sbuf, rbuf, op=op)
        finally:
            op.Free()

    def _find_closest_node(self, pts):
        comm, rank, root = get_comm_rank_root()

        # Read our portion of the nodes table
        start, end, _ = get_start_end_csize(comm, len(self.mesh.raw['nodes']))
        nodes = self.mesh.raw['nodes'][start:end]

        # Insert these points into a spatial index
        props = Property(dimension=self.mesh.ndims, interleaved=True)
        ins = ((i, [*p, *p], None) for i, p in enumerate(nodes))
        idx = Index(ins, properties=props)

        # Query the index to find our closest node
        nearest = np.array([next(idx.nearest([*p, *p], 1)) for p in pts])

        buf = np.empty(len(pts), dtype=[('dist', float), ('idx', int)])
        buf['dist'] = np.linalg.norm(pts - nodes[nearest], axis=1)
        buf['idx'] = nearest + start

        self._minloc(comm.Allreduce, mpi.IN_PLACE, buf)

        return buf

    def _find_closest_element(self, etype, pts, nearest):
        spts = self.mesh.spts[etype]
        nodes = self.mesh.spts_nodes[etype]

        # See which of our elements contain the nearest node
        eidx, nidx = np.isin(nodes, nearest['idx']).nonzero()

        # Create a map from node number to element indices
        neles = defaultdict(set)
        for ei, ni in zip(eidx, nodes[eidx, nidx]):
            neles[ni].add(ei)

        # Use this to form the set of candidate elements for each point
        pidx, sidx = [], []
        for i, (di, ni) in enumerate(nearest):
            for ei in neles.get(ni, []):
                pidx.append(i)
                sidx.append(ei)

        # Obtain the closest location inside each of these elements
        dists, tlocs = self._compute_tlocs(etype, spts[:, sidx], pts[pidx])

        # For each query point identify the most promising element
        closest = {}
        for i, (pi, dist, tloc) in enumerate(zip(pidx, dists, tlocs)):
            if pi not in closest or dist < closest[pi][0]:
                closest[pi] = (dist, i)

        pidx = list(closest)
        tidx = [i for d, i in closest.values()]
        sidx = [sidx[i] for i in tidx]

        return dict(zip(pidx, zip(dists[tidx], sidx, tlocs[tidx])))

    def _initial_tlocs(self, etype, spts, plocs):
        shape, basis, order = self._get_shape_basis_order(etype, len(spts))

        # Obtain a fine sampling of points inside each element
        fop = basis.nodal_basis_at(shape.std_ele(self.fine_order))
        fpts = fop @ spts.reshape(len(spts), -1)
        fpts = fpts.reshape(len(fop), *spts.shape[1:])

        # Find the closest fine sample point to each query point
        dists = np.linalg.norm(fpts - plocs, axis=2)
        amins = np.unravel_index(dists.argmin(axis=0), fpts.shape[:2])

        return fpts[amins]

    def _compute_tlocs(self, etype, spts, plocs):
        shape, basis, order = self._get_shape_basis_order(etype, len(spts))

        # Evaluate the initial guesses
        ktlocs = self._initial_tlocs(etype, spts, plocs)
        kplocs = np.einsum('ij,jik->ik',
                           basis.nodal_basis_at(ktlocs, clean=False), spts)

        # Apply three iterations of Newton's method
        for k in range(3):
            jac_ops = basis.jac_nodal_basis_at(ktlocs, clean=False)

            A = np.einsum('ijk,jkl->kli', jac_ops, spts)
            b = kplocs - plocs
            ktlocs -= np.linalg.solve(A, b[..., None]).squeeze()

            ops = basis.nodal_basis_at(ktlocs, clean=False)
            np.einsum('ij,jik->ik', ops, spts, out=kplocs)

        # Compute the final distances
        dists = np.linalg.norm(kplocs - plocs, axis=1)

        # Prune invalid points
        for i, t in enumerate(ktlocs):
            if not shape.valid_spt(t):
                dists[i] = np.inf

        return dists, ktlocs


class SamplerPlugin(BaseSolnPlugin):
    name = 'sampler'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Output frequency and format
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')

        # Search locations in transformed and physical space
        self.pts, self._pinfo = self._process_pts(intg)

        # Tell the root rank which points we are responsible for
        ptsrank = comm.gather(list(self._pinfo), root=root)

        if rank == root:
            nvars = self.nvars

            # Allocate a buffer to store the sampled points
            self._ptsbuf = ptsbuf = np.empty((len(self.pts), self.nvars))

            # Compute the counts and displacements, sans nvars
            ptscounts = np.array([len(pr) for pr in ptsrank])
            ptsdisps = np.concatenate(([0], np.cumsum(ptscounts[:-1])))

            # Form the MPI Gatherv receive buffer tuple
            self._ptsrecv = (ptsbuf, (nvars*ptscounts, nvars*ptsdisps))

            # Form the reordering list
            self._ptsinv = np.argsort([i for pr in ptsrank for i in pr])

            # Open the output file
            self.outf = init_csv(self.cfg, cfgsect, self._header)
        else:
            self._ptsrecv = None

    @property
    def _header(self):
        colnames = ['t', 'x', 'y', 'z'][:self.ndims + 1]

        if self.fmt == 'primitive':
            colnames += self.elementscls.privarmap[self.ndims]
        else:
            colnames += self.elementscls.convarmap[self.ndims]

        return ','.join(colnames)

    def _load_pts_locs(self, mesh):
        spts = self.cfg.get(self.cfgsect, 'samp-pts')

        # If we have a named set then load the points from the mesh
        if re.match(r'\w+$', spts):
            comm, rank, root = get_comm_rank_root()

            if rank == root:
                sinfo = mesh.raw[f'plugins/sampler/{spts}'][:]
            else:
                sinfo = None

            sinfo = comm.bcast(sinfo, root=root)

            return sinfo['ploc'], sinfo[['cidx', 'eidx', 'tloc']]
        # Otherwise we have a list of points to parse and locate
        else:
            pts = np.array(self.cfg.getliteral(self.cfgsect, 'samp-pts'))
            locs = _PointLocator(mesh).locate(pts)

            return pts, locs[['cidx', 'eidx', 'tloc']]

    def _process_pts(self, intg):
        mesh = intg.system.mesh
        pts, locs = self._load_pts_locs(mesh)

        pinfo = {}

        for j, (etype, eidxs) in enumerate(mesh.eidxs.items()):
            eimap = np.argsort(eidxs)
            ubasis = intg.system.ele_map[etype].basis.ubasis

            # Filter points which do not belong to this element type
            elocs = locs['cidx'] == mesh.codec.index(f'eles/{etype}')
            elocs = elocs.nonzero()[0]

            # See what points we have
            esrch = np.searchsorted(eidxs, locs[elocs]['eidx'], sorter=eimap)
            for i, k, l in zip(elocs, esrch, locs[elocs]):
                if k < eimap.size and eidxs[eimap[k]] == l['eidx']:
                    op = ubasis.nodal_basis_at(l['tloc'][None], clean=False)
                    pinfo[i] = (j, eimap[k], op)

        return pts, pinfo

    def _process_samples(self, samps):
        samps = np.array(samps)

        # If necessary then convert to primitive form
        if self.fmt == 'primitive' and samps.size:
            samps = self.elementscls.con_to_pri(samps.T, self.cfg)
            samps = np.array(samps).T

        return np.ascontiguousarray(samps, dtype=float)

    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the solution matrices
        solns = intg.soln

        # Get information about the points we are responsible for
        pinfo = self._pinfo.values()

        # Perform the sampling and interpolation
        samples = [op @ solns[et][:, :, ei] for et, ei, op in pinfo]
        samples = self._process_samples(samples)

        # Gather to the root rank
        comm.Gatherv(samples, self._ptsrecv, root=root)

        # If we're the root rank then output
        if rank == root:
            for ploc, samp in zip(self.pts, self._ptsbuf[self._ptsinv]):
                print(intg.tcurr, *ploc, *samp, sep=',', file=self.outf)

            # Flush to disk
            self.outf.flush()


class SamplerCLIPlugin(BaseCLIPlugin):
    name = 'sampler'

    @classmethod
    def add_cli(cls, parser):
        sp = parser.add_subparsers()

        # Add command
        ap_add = sp.add_parser('add', help='sampler add --help')
        ap_add.add_argument('mesh', help='input mesh file')
        ap_add.add_argument('pts', type=FileType('r'),
                            help='input points file')
        ap_add.add_argument('name', nargs='?', help='point set name')
        ap_add.add_argument('-f', '--force', action='count',
                            help='overwrite existing point set')
        ap_add.add_argument('--skip', type=int, default=0,
                            help='number of rows to skip')
        ap_add.set_defaults(process=cls.add_cmd)

        # List command
        ap_list = sp.add_parser('list', help='sampler list --help')
        ap_list.add_argument('mesh', help='input mesh file')
        ap_list.add_argument('-s', '--sep', default='\t', help='separator')
        ap_list.set_defaults(process=cls.list_cmd)

        # Dump command
        ap_dump = sp.add_parser('dump', help='sampler dump --help')
        ap_dump.add_argument('mesh', help='input mesh file')
        ap_dump.add_argument('name', help='point set')
        ap_dump.add_argument('-s', '--sep', default='\t', help='separator')
        ap_dump.set_defaults(process=cls.dump_cmd)

        # Remove command
        ap_remove = sp.add_parser('remove', help='sampler remove --help')
        ap_remove.add_argument('mesh', help='input mesh file')
        ap_remove.add_argument('name', help='point set')
        ap_remove.set_defaults(process=cls.remove_cmd)

    @cli_external
    def add_cmd(self, args):
        # Initialise MPI
        init_mpi()

        # Get our MPI info
        comm, rank, root = get_comm_rank_root()

        # Read the mesh
        reader = NativeReader(args.mesh, construct_con=False)
        mesh = reader.mesh

        if rank == root:
            # Get the point set name
            pname = args.name or Path(args.pts.name).stem
            if not re.match(r'\w+$', pname):
                raise ValueError('Invalid point set name')

            # Check it does not already exist unless --force is given
            if f'plugins/sampler/{pname}' in mesh.raw and not args.force:
                raise ValueError(f'Point set {pname} already exists; use '
                                    '-f to replace')

            def line_reader(f):
                for l in f:
                    yield l.replace(',', ' ')

            # Read in the points
            pts = np.loadtxt(line_reader(args.pts), skiprows=args.skip)

            # Validate the dimensionality
            if pts.shape[1] != mesh.ndims:
                raise ValueError('Invalid point set dimensionality')
        else:
            pts = None

        # Broadcast the points
        pts = comm.bcast(pts, root=root)

        # Identify which element each point is located in
        locs = _PointLocator(mesh).locate(pts)

        # Close the mesh file so it can be reopened for writing
        reader.close()

        # Have the root rank write the point and location data out
        if rank == root:
            dtype = [('ploc', float, mesh.ndims), ('cidx', np.int16),
                     ('eidx', np.int64), ('tloc', float, mesh.ndims)]
            sinfo = np.empty(len(pts), dtype=dtype)
            sinfo['ploc'] = pts
            sinfo[['cidx', 'eidx', 'tloc']] = locs[['cidx', 'eidx', 'tloc']]

            with h5py.File(args.mesh, 'r+') as f:
                g = f.require_group('plugins/sampler')

                # Remove any existing sample point info
                if pname in g:
                    del g[pname]

                # Save the sample point info
                g[pname] = sinfo

    @cli_external
    def list_cmd(self, args):
        with h5py.File(args.mesh, 'r') as mesh:
            g = mesh.require_group('plugins/sampler')

            print('name', 'npts', sep=args.sep)
            for name, points in sorted(g.items()):
                print(name, len(points), sep=args.sep)

    @cli_external
    def dump_cmd(self, args):
        with h5py.File(args.mesh, 'r') as mesh:
            points = mesh[f'plugins/sampler/{args.name}']['ploc']
            ndim = points.shape[1]

            print(*'xyz'[:ndim], sep=args.sep)
            for p in points:
                print(*p, sep=args.sep)

    @cli_external
    def remove_cmd(self, args):
        with h5py.File(args.mesh, 'r+') as mesh:
            if args.name not in mesh['plugins/sampler']:
                raise ValueError(f'Point set {args.name} does not exist')

            del mesh[f'plugins/sampler/{args.name}']
