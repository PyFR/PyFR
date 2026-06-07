from argparse import FileType
import csv
import io
from pathlib import Path
import re

import h5py
import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external
from pyfr.plugins.fields.runner import FieldRunner
from pyfr.points import PointLocator
from pyfr.readers.native import NativeReader
from pyfr.snapshot import from_file as snap_from_file


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
        ap_sample.add_argument(
            '--add-fields', dest='add_fields', action='append', default=[],
            metavar='NAME[,NAME,...]',
            help='register derived-field providers (mach, yplus, cf, ...); '
            'may be repeated, comma lists ok'
        )
        ap_sample.add_argument(
            '--remove-fields', dest='remove_fields', action='append',
            default=[], metavar='NAME[,NAME,...]',
            help='drop these column names from the CSV; may be repeated, '
            'comma lists ok'
        )
        ap_sample.add_argument('--cfg', dest='field_cfg',
                               help='config file for field providers')
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
        comm, rank, root = get_comm_rank_root()

        # Flatten comma-separated --add-fields / --remove-fields
        add_fields = [n.strip() for a in args.add_fields
                      for n in a.split(',') if n.strip()]
        remove_fields = {n.strip() for a in args.remove_fields
                         for n in a.split(',') if n.strip()}

        # Read the mesh and solution
        snap = snap_from_file(args.mesh, args.soln, args.pname)
        mesh = snap.mesh

        # Dimension and field names
        dims = 'xyz'[:mesh.ndims]

        # Soln-form-only flags
        if args.format == 'primitive' and snap.prefix != 'soln':
            raise ValueError('Primitive output only supported for conservative'
                             f'-form solution files (snap.prefix = '
                             f'{snap.prefix!r})')
        if add_fields and snap.prefix != 'soln':
            raise ValueError('Field providers require conservative-form '
                             f'solution data (snap.prefix = {snap.prefix!r})')

        # Resolve the points (either from a CSV file or a pre-stored set)
        if args.pts:
            # Read the sample points from a CSV file
            pts = (_read_pts(args.pts, ndims=mesh.ndims, skip=args.skip)
                   if rank == root else None)
        else:
            # Obtain the pre-processed sample points from the mesh
            pdata = (mesh.raw[f'plugins/sampler/{args.name}'][:]
                     if rank == root else None)
            pdata = comm.bcast(pdata, root=root)
            pts = pdata['ploc']
        pts = comm.bcast(pts, root=root) if args.pts else pts

        # Construct and configure the point sampler
        region = snap.at_points(pts)

        # Sample the solution
        sample = region.sample(snap)
        has_grads = snap.has_grads

        # Resolve field providers and run them on the sample
        field_cfg = (Inifile.load(args.field_cfg) if args.field_cfg
                     else snap.cfg)
        runner = FieldRunner(add_fields, mesh.ndims, field_cfg, 'volume')
        sample.run(runner, public_only=True)

        # Have the root rank post-process and write the samples
        if rank != root:
            return

        # Build the output as an ordered dict
        fields = {}

        if args.format == 'primitive':
            for info in snap.iter_fields():
                if info.source not in ('primitive', 'gradient'):
                    continue
                if info.source == 'gradient' and not has_grads:
                    continue
                arr = sample.field_array('points', info)
                for cname, col in zip(info.components, arr.T):
                    if info.source == 'primitive':
                        fields[cname] = col
                    else:
                        var, _, d = cname.rpartition('-')
                        fields[f'grad_{var}_{dims[int(d)]}'] = col
        else:
            data_infos = sorted(
                (f for f in snap.fields.values() if f.source == 'data'),
                key=lambda f: f.data_index)
            for info in data_infos:
                fields[info.name] = sample.field_array('points', info)
            if has_grads:
                grad_infos = sorted(
                    (f for f in snap.fields.values() if f.source == 'grad_data'),
                    key=lambda f: f.data_index)
                for info in grad_infos:
                    var = info.name.removeprefix('grad ')
                    arr = sample.field_array('points', info)
                    for d, dim in enumerate(dims):
                        fields[f'grad_{var}_{dim}'] = arr[:, d]

        # Field provider outputs
        for fname, varnames in runner.fields(public_only=True).items():
            arr = sample.field_arrays.get(('points', fname))
            if arr is None:
                continue
            if len(varnames) == 1:
                fields[varnames[0]] = arr if arr.ndim == 1 else arr.ravel()
            else:
                for i, vn in enumerate(varnames):
                    fields[vn] = arr[:, i] if arr.ndim == 2 else arr[i]

        for n in remove_fields:
            fields.pop(n)

        # Write out the header
        print(*dims, *fields, sep=args.sep)

        # Write out the samples
        for i, ploc in enumerate(pts):
            print(*ploc, *(arr[i] for arr in fields.values()), sep=args.sep)
