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
from pyfr.plugins.postproc.adapters import PostProcData
from pyfr.plugins.postproc.base import get_pp_plugins
from pyfr.points import PointLocator, PointSampler
from pyfr.readers.native import NativeReader
from pyfr.util import subclass_where


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


def _process_con_to_pri(elementscls, ndims, cfg, *, has_grads=False):
    nvars = len(elementscls.convars(ndims, cfg))
    con_to_pri = elementscls.con_to_pri
    diff_con_to_pri = elementscls.diff_con_to_pri

    def process(samps):
        if samps.size:
            samps = samps.T

            # Convert the samples to primitive variables
            psamps = con_to_pri(samps[:nvars], cfg)

            # Also convert any gradient data
            if has_grads:
                diff_con = samps[nvars:].reshape(nvars, ndims, -1)
                diff_pri = diff_con_to_pri(samps[:nvars], diff_con, cfg)

                psamps += [f for gf in diff_pri for f in gf]

            samps = np.array(psamps).T

        return samps

    return process


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
            '--postproc', dest='pp_plugins', action='append', default=[],
            metavar='PLUGIN', help='postprocessing plugin; may be repeated'
        )
        ap_sample.add_argument('--cfg', dest='pp_cfg',
                               help='config file for postproc plugins')
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
        fields = soln.fields

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

        # Determine if gradient data is present
        has_grads = bool(soln.grad_data)

        # If gradients exist, stack them into the solution data
        sdata = []
        for etype in mesh.eidxs:
            d = soln.data[etype]
            if has_grads:
                g = soln.grad_data[etype].transpose(1, 2, 0, 3)
                g = g.reshape(g.shape[0], -1, g.shape[3])
                d = np.concatenate([d, g], axis=1)

            sdata.append(d)

        # Postproc plugins require primitive format
        if args.pp_plugins and args.format != 'primitive':
            raise ValueError('Postproc plugins require --format primitive')

        # Handle conversion from conservative to primitive variables
        if args.format == 'primitive':
            from pyfr.solvers.base import BaseSystem

            if soln.stats.get('data', 'prefix') != 'soln':
                raise ValueError('Primitive output only supported for '
                                 'solution files')

            # Obtain the system associated with the solution
            systemcls = subclass_where(
                BaseSystem, name=soln.config.get('solver', 'system')
            )
            elementscls = systemcls.elementscls
            vmap = elementscls.privars(mesh.ndims, soln.config)

            fields = list(vmap)
            if has_grads:
                fields.extend(f'grad_{v}_{d}' for v in vmap for d in dims)

            process = _process_con_to_pri(elementscls, mesh.ndims,
                                          soln.config, has_grads=has_grads)
        else:
            process = None
            if has_grads:
                fields = list(fields)
                fields.extend(f'grad_{v}_{d}'
                              for v in soln.fields for d in dims)

        # Resolve postproc plugins + dependencies in topological order
        pp_cfg = Inifile.load(args.pp_cfg) if args.pp_cfg else soln.config
        pp_plugins = get_pp_plugins(args.pp_plugins, mesh.ndims, pp_cfg,
                                    'volume')

        # Construct and configure the point sampler
        sampler = PointSampler(mesh, spts)
        sampler.configure_with_cfg_nvars(soln.config, len(fields))

        # Sample the solution
        samps = sampler.sample(sdata, process=process)

        # Run postproc plugins at the sampled points (primitive only)
        if rank == root:
            adapter = PostProcData(soln.config, soln, samps.T, pts.T)
            pp_field_map = {}
            for pp in pp_plugins:
                pp.run(adapter)
                pp_field_map.update(pp.fields())

            extra_cols = []
            for name, arr in adapter.fields.items():
                if name.startswith('_'):
                    continue
                fields.extend(pp_field_map[name])
                extra_cols.append(np.atleast_2d(arr.T).T)
            samps = np.concatenate([samps, *extra_cols], axis=1)

        if rank == root:
            print(*dims, *fields, sep=args.sep)

            for ploc, samp in zip(pts, samps):
                print(*ploc, *samp, sep=args.sep)
