from argparse import FileType
from collections import defaultdict
from pathlib import Path
import re

import h5py
import numpy as np
from rtree.index import Index, Property

from pyfr.mpiutil import get_comm_rank_root, get_start_end_csize, init_mpi, mpi
from pyfr.nputil import iter_struct
from pyfr.plugins.base import (BaseCLIPlugin, BaseSolnPlugin, DatasetAppender,
                               cli_external, init_csv, open_hdf5_a)
from pyfr.polys import get_polybasis
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.util import first, memoize, subclass_where


class PointLocator:
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


class PointSampler:
    def __init__(self, mesh, spts):
        self.mesh = mesh

        # If spts is a string then treat it as a named point set
        if isinstance(spts, str):
            comm, rank, root = get_comm_rank_root()

            if rank == root:
                sinfo = mesh.raw[f'plugins/sampler/{spts}'][:]
            else:
                sinfo = None

            sinfo = comm.bcast(sinfo, root=root)

            pts, locs = sinfo['ploc'], sinfo[['cidx', 'eidx', 'tloc']]
        # Otherwise, treat it as a list of points
        else:
            pts = np.array(spts)
            locs = PointLocator(mesh).locate(pts)[['cidx', 'eidx', 'tloc']]

        self.pts, self.locs = pts, locs

    def configure_with_intg_nvars(self, intg, nvars):
        # Get the solution bases from the system
        ubases = {etype: eles.basis.ubasis
                  for etype, eles in intg.system.ele_map.items()}

        self._configure_ubases_nvars(ubases, nvars)

    def configure_with_cfg_nvars(self, cfg, nvars):
        ubases = {}

        for etype in self.mesh.eidxs:
            shapecls = subclass_where(BaseShape, name=etype)
            ubases[etype] = shapecls(None, cfg).ubasis

        self._configure_ubases_nvars(ubases, nvars)

    def _configure_ubases_nvars(self, ubases, nvars):
        self.nvars = nvars
        locs = self.locs

        comm, rank, root = get_comm_rank_root()
        ptsrank, pinfo = [], defaultdict(list)

        for j, (etype, eidxs) in enumerate(self.mesh.eidxs.items()):
            eimap = np.argsort(eidxs)
            ubasis = ubases[etype]

            # Filter points which do not belong to this element type
            elocs = locs['cidx'] == self.mesh.codec.index(f'eles/{etype}')
            elocs = elocs.nonzero()[0]

            # See what points we have
            esrch = np.searchsorted(eidxs, locs[elocs]['eidx'], sorter=eimap)
            for i, k, l in zip(elocs, esrch, locs[elocs]):
                if k < eimap.size and eidxs[eimap[k]] == l['eidx']:
                    op = ubasis.nodal_basis_at(l['tloc'][None], clean=False)

                    ptsrank.append(i)
                    pinfo[j, eimap[k]].append((i, op))

        # Group points according to the element they're inside
        self.pinfo, self.pcount = [], len(ptsrank)
        for (et, ei), info in pinfo.items():
            if len(info) == 1:
                self.pinfo.append((et, ei, *info[0]))
            else:
                idxs, ops = zip(*info)
                self.pinfo.append((et, ei, np.array(idxs), np.vstack(ops)))

        # Tell the root rank which points we are responsible for
        ptsrank = comm.gather(ptsrank, root=root)

        if rank == root:
            # Allocate a buffer to store the sampled points
            self._ptsbuf = ptsbuf = np.empty((len(self.pts), nvars))

            # Compute the counts and displacements, sans nvars
            ptscounts = np.array([len(pr) for pr in ptsrank])
            ptsdisps = np.concatenate(([0], np.cumsum(ptscounts[:-1])))

            if ptscounts.sum() != len(self.pts):
                raise RuntimeError('Missing points in solution')

            # Form the MPI Gatherv receive buffer tuple
            self._ptsrecv = (ptsbuf, (nvars*ptscounts, nvars*ptsdisps))

            # Form the reordering list
            self._ptsinv = np.argsort([i for pr in ptsrank for i in pr])

    def sample(self, solns, process=None):
        comm, rank, root = get_comm_rank_root()

        # Perform the sampling
        samples = np.empty((self.pcount, self.nvars))
        for et, ei, idxs, ops in self.pinfo:
            samples[idxs] = ops @ solns[et][:, :, ei]

        # Post-process the samples
        if process:
            samples = np.ascontiguousarray(process(samples))

        # Gather to the root rank and return
        if rank == root:
            comm.Gatherv(samples, self._ptsrecv, root=root)
            return self._ptsbuf[self._ptsinv]
        else:
            comm.Gatherv(samples, None, root=root)
            return None


def _read_pts(ptsf, ndims=None, skip=0):
    def line_reader(f):
        for l in f:
            yield l.replace(',', ' ')

    # Read in the points
    pts = np.loadtxt(line_reader(ptsf), skiprows=skip)
    pts = np.atleast_2d(pts)

    # Validate the dimensionality
    if ndims and pts.shape[1] != ndims:
        raise ValueError('Invalid point set dimensionality')

    return pts


def _process_con_to_pri(elementscls, ndims, cfg):
    nvars = len(elementscls.convars(ndims, cfg))
    con_to_pri = elementscls.con_to_pri
    diff_con_to_pri = elementscls.diff_con_to_pri

    def process(samps):
        if samps.size:
            samps = samps.T

            # Convert the samples to primitive variables
            psamps = con_to_pri(samps[:nvars], cfg)

            # Also convert any gradient data
            if len(samps) == (1 + ndims)*nvars:
                diff_con = samps[nvars:].reshape(nvars, ndims, -1)
                diff_pri = diff_con_to_pri(samps[:nvars], diff_con, cfg)

                psamps += [f for gf in diff_pri for f in gf]

            samps = np.array(psamps).T

        return samps

    return process


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

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Output format
        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')
        if self.fmt == 'primitive':
            self._process = _process_con_to_pri(self.elementscls, self.ndims,
                                                self.cfg)
        else:
            self._process = None

        # Decide if gradients should be sampled or not
        self._sample_grads = self.cfg.getbool(cfgsect, 'sample-gradients',
                                              False)

        # Get the sample points
        spts = self.cfg.get(self.cfgsect, 'samp-pts')
        if ',' in spts:
            spts = self.cfg.getliteral(self.cfgsect, 'samp-pts')

        # Determine the number of variables per sample
        self.nsvars = (1 + self.ndims*self._sample_grads)*self.nvars

        # Construct and configure the point sampler
        self.psampler = PointSampler(intg.system.mesh, spts)
        self.psampler.configure_with_intg_nvars(intg, self.nsvars)

        # Have the root rank open the output file
        if rank == root:
            match self.cfg.get(cfgsect, 'file-format', 'csv'):
                case 'csv':
                    self._init_csv(intg)
                case 'hdf5':
                    self._init_hdf5(intg)
                case _:
                    raise ValueError('Invalid file format')

    def _init_csv(self, intg):
        self.outf = init_csv(self.cfg, self.cfgsect, self._header(intg))
        self._write = self._write_csv

    def _write_csv(self, t, samps):
        for ploc, samp in zip(self.psampler.pts, samps):
            print(t, *ploc, *samp, sep=',', file=self.outf)

        # Flush to disk
        self.outf.flush()

    def _init_hdf5(self, intg):
        outf = open_hdf5_a(self.cfg.get(self.cfgsect, 'file'))

        pts = self.psampler.pts
        npts, nsvars = len(pts), self.nsvars
        chunk = 128

        dset = self.cfg.get(self.cfgsect, 'file-dataset')
        if dset in outf:
            d = outf[dset]
            t = d.dims[0][0]
            p = d.dims[1][0]

            # Ensure the point sets are compatible
            if p.shape != pts.shape or not np.allclose(p[:], pts):
                raise ValueError('Inconsistent sample points')
        else:
            d = outf.create_dataset(dset, (0, npts, nsvars), float,
                                    chunks=(chunk, min(4, npts), nsvars),
                                    maxshape=(None, npts, nsvars))
            t = outf.create_dataset(f'{dset}_t', (0,), float, chunks=(chunk,),
                                    maxshape=(None,))
            p = outf.create_dataset(f'{dset}_p', data=pts)

            t.make_scale('t')
            p.make_scale('pts')
            d.dims[0].attach_scale(t)
            d.dims[1].attach_scale(p)

        self._t = DatasetAppender(t)
        self._samps = DatasetAppender(d)
        self._write = self._write_hdf5

    def _write_hdf5(self, t, samps):
        self._t(t)
        self._samps(samps)

    def _header(self, intg):
        eles = first(intg.system.ele_map.values())
        dims = 'xyz'[:self.ndims]

        vmap = eles.privars if self.fmt == 'primitive' else eles.convars

        colnames = ['t', *dims, *vmap]
        if self._sample_grads:
            colnames.extend(f'grad_{v}_{d}' for v in vmap for d in dims)

        return ','.join(colnames)

    def _process_pri(self, samps):
        if samps.size:
            ecls = self.elementscls
            samps = samps.T

            # Convert the samples to primitive variables
            psamps = ecls.con_to_pri(samps[:self.nvars], self.cfg)

            # Also convert any gradient data
            if self._sample_grads:
                diff_con = samps[nvars:].reshape(self.nvars, self.ndims, -1)
                psamps += ecls.diff_con_to_pri(samps[:self.nvars], diff_con,
                                               self.cfg)

            samps = np.array(psamps).T

        return samps

    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps % self.nsteps:
            return

        # Fetch the solution
        soln = list(intg.soln)

        # If requested also fetch solution gradients
        if self._sample_grads:
            for i, g in enumerate(intg.grad_soln):
                g = g.transpose(1, 2, 0, 3)
                g = g.reshape(len(g), self.ndims*self.nvars, -1)
                soln[i] = np.hstack([soln[i], g])

        # Perform the sampling
        samps = self.psampler.sample(soln, process=self._process)

        # If we're the root rank then output
        if samps is not None:
            self._write(intg.tcurr, samps)


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

        # Sample command
        ap_sample = sp.add_parser('sample', help='sampler sample --help')
        ap_sample.add_argument('mesh', help='input mesh file')
        ap_sample.add_argument('soln', help='input solution file')
        sample_opts = ap_sample.add_mutually_exclusive_group(required=True)
        sample_opts.add_argument('-n', '--name', help='point set')
        sample_opts.add_argument('-p', '--pts', type=FileType('r'),
                                 help='input points file')
        ap_sample.add_argument('--skip', type=int, default=0,
                               help='number of rows to skip')
        ap_sample.add_argument(
            '-f', '--format',  choices=['conservative', 'primitive'],
             default='conservative', help='output format'
        )
        ap_sample.add_argument('-s', '--sep', default='\t', help='separator')
        ap_sample.set_defaults(process=cls.sample_cmd)

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

            pts = _read_pts(args.pts, ndims=mesh.ndims, skip=args.skip)
        else:
            pts = None

        # Broadcast the points
        pts = comm.bcast(pts, root=root)

        # Identify which element each point is located in
        locs = PointLocator(mesh).locate(pts)

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

    @cli_external
    def sample_cmd(self, args):
        # Initialise MPI
        init_mpi()

        # Get our MPI info
        comm, rank, root = get_comm_rank_root()

        # Read the mesh and solution
        reader = NativeReader(args.mesh, construct_con=False)
        mesh, soln = reader.load_subset_mesh_soln(args.soln)

        # Dimension and field names
        dims = 'xyz'[:mesh.ndims]
        fields = soln['stats'].get('data', 'fields').split(',')

        # Read the sample points from a CSV file
        if args.pts:
            if rank == root:
                pts = _read_pts(args.pts, ndims=mesh.ndims, skip=args.skip)
            else:
                pts = None

            spts = pts = comm.bcast(pts, root=root)
        # Obtain the pre-processed sample points from the mesh
        else:
            spts = args.name

            if rank == root:
                pts = mesh.raw[f'plugin/sampler/{spts}']['ploc']

        # Handle conversion from conservative to primitive variables
        if args.format == 'primitive':
            from pyfr.solvers.base import BaseSystem

            if soln['stats'].get('data', 'prefix') != 'soln':
                raise ValueError('Primitive output only supported for '
                                 'solution files')

            # Obtain the system associated with the solution
            systemcls = subclass_where(
                BaseSystem, name=soln['config'].get('solver', 'system')
            )
            elementscls = systemcls.elementscls
            vmap = elementscls.privars(mesh.ndims, soln['config'])

            # Solution data
            if len(fields) == len(vmap):
                fields = list(vmap)
            # Solution and gradient data
            elif len(fields) == (1 + mesh.ndims)*len(vmap):
                fields = list(vmap)
                fields.extend(f'grad_{v}_{d}' for v in vmap for d in dims)
            else:
                raise ValueError('Invalid number of field variables')

            process = _process_con_to_pri(elementscls, mesh.ndims,
                                          soln['config'])
        else:
            process = None

        # Construct and configure the point sampler
        sampler = PointSampler(mesh, spts)
        sampler.configure_with_cfg_nvars(soln['config'], len(fields))

        # Sample the solution
        samps = sampler.sample([soln[etype] for etype in mesh.eidxs],
                               process=process)

        if rank == root:
            print(*dims, *fields, sep=args.sep)

            for ploc, samp in zip(pts, samps):
                print(*ploc, *samp, sep=args.sep)
