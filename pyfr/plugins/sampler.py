from argparse import FileType
import csv
import io
from pathlib import Path
import re

import h5py
import numpy as np

from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.plugins.base import (BaseCLIPlugin, BaseSolnPlugin, DatasetAppender,
                               cli_external, init_csv, open_hdf5_a)
from pyfr.points import PointLocator, PointSampler
from pyfr.readers.native import NativeReader
from pyfr.util import first, subclass_where


def _read_pts(ptsf, ndims=None, skip=0):
    # Read the points
    pts = ''.join(list(ptsf)[skip:])

    # Parse them
    dialect = csv.Sniffer().sniff(pts)
    pts = csv.reader(io.StringIO(pts), dialect=dialect)
    pts = np.array([[float(f) for f in p] for p in pts if p])

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
        ap_add.add_argument('-P', '--pname', help='partitioning to use')
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
        ap_sample.add_argument('-P', '--pname', help='partitioning to use')
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
        reader = NativeReader(args.mesh, args.pname, construct_con=False)
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
            sgroup = mesh.get('plugins/sampler')

            if sgroup is None or args.name not in sgroup:
                raise ValueError(f'Point set {args.name} does not exist')

            del sgroup[args.name]

    @cli_external
    def sample_cmd(self, args):
        # Initialise MPI
        init_mpi()

        # Get our MPI info
        comm, rank, root = get_comm_rank_root()

        # Read the mesh and solution
        reader = NativeReader(args.mesh, args.pname, construct_con=False)
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
