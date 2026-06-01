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
from pyfr.points import PointLocator, PointSampler
from pyfr.readers.native import NativeReader


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

        # Field providers require primitive format (provider deps + outputs)
        if add_fields and args.format != 'primitive':
            raise ValueError('Field providers require --format=primitive')

        # Soln path goes through snap.at_points (region samples + provides
        # con->pri + grad->grad-pri + postproc).  Scalar / non-soln files
        # (tavg, residual) still use the raw PointSampler path below.
        from pyfr.snapshot import FileSnapshot

        snap = FileSnapshot(args.mesh, args.soln, args.pname)
        mesh = snap.mesh
        dims = 'xyz'[:mesh.ndims]
        is_soln = snap.stats.get('data', 'prefix') == 'soln'

        if args.format == 'primitive' and not is_soln:
            raise ValueError('Primitive output only supported for solution '
                             'files')

        # Resolve the points (either from a CSV file or a pre-stored set)
        if args.pts:
            pts = (_read_pts(args.pts, ndims=mesh.ndims, skip=args.skip)
                   if rank == root else None)
        else:
            pdata = (mesh.raw[f'plugins/sampler/{args.name}'][:]
                     if rank == root else None)
            pdata = comm.bcast(pdata, root=root)
            pts = pdata['ploc']
        pts = comm.bcast(pts, root=root) if args.pts else pts

        if is_soln:
            # snap.at_points internally builds PointLocator + PointSampler
            region = snap.at_points(pts)
            sample = region.sample(snap)
            has_grads = snap.has_grads

            privars = snap.elementscls.privars(mesh.ndims, snap.config)
            if args.format == 'primitive':
                col_names = list(privars)
                if has_grads:
                    col_names.extend(f'grad_{v}_{d}' for v in privars
                                     for d in dims)
            else:
                col_names = list(snap.stored_fields)
                if has_grads:
                    col_names.extend(f'grad_{v}_{d}' for v in snap.stored_fields
                                     for d in dims)

            # Run field providers on the sample (results land in sample.fields).
            # field_cfg overrides snap.config when --cfg is supplied.
            field_cfg = (Inifile.load(args.field_cfg) if args.field_cfg
                         else snap.config)
            if add_fields:
                sample.run(add_fields, public_only=True, cfg=field_cfg)

            if rank != root:
                return

            if args.format == 'primitive':
                # Stack pris + grad_pris components into (npts, nfields)
                pris_cols = [p[:, None] for p in sample.pris['points']]
                cols = pris_cols
                if has_grads and sample.grad_pris['points'] is not None:
                    for g in sample.grad_pris['points']:
                        cols.extend(g[d][:, None] for d in range(mesh.ndims))
                samps = np.hstack(cols)
            else:
                samps = sample.samples

            # Append derived fields (resolved via runner.fields() ordering)
            if add_fields:
                runner = FieldRunner(add_fields, mesh.ndims, field_cfg,
                                     'volume')
                extra = []
                for name, varnames in runner.fields(public_only=True).items():
                    col_names.extend(varnames)
                    arr = sample.fields.get(('points', name))
                    if arr is None:
                        continue
                    extra.append(arr[:, None] if arr.ndim == 1 else arr.T)
                if extra:
                    samps = np.hstack([samps, *extra])
        else:
            # Non-soln (tavg / residual / ...): raw PointSampler path; no
            # field providers by construction (format != 'primitive' check
            # above already rejected them).
            has_grads = snap.has_grads
            col_names = list(snap.stored_fields)
            if has_grads:
                col_names.extend(f'grad_{v}_{d}' for v in snap.stored_fields
                                 for d in dims)

            sdata = []
            for etype in mesh.eidxs:
                d = snap.soln(etype)
                if has_grads:
                    g = snap.grad_soln(etype).transpose(1, 2, 0, 3)
                    g = g.reshape(g.shape[0], -1, g.shape[3])
                    d = np.concatenate([d, g], axis=1)
                sdata.append(d)

            locs = (pdata[['cidx', 'eidx', 'tloc']]
                    if not args.pts else None)
            sampler = PointSampler(mesh, pts, locs)
            sampler.configure_with_cfg_nvars(snap.config, len(col_names))
            samps = sampler.sample(sdata)
            if rank != root:
                return

        # Apply --remove-fields trim to columns (after providers + grads have
        # been appended to col_names).  Catch typos with a strict check.
        if remove_fields:
            unknown = remove_fields - set(col_names)
            if unknown:
                raise ValueError(
                    f'--remove-fields names not in output: {sorted(unknown)} '
                    f'(available: {col_names})'
                )
            keep = [i for i, n in enumerate(col_names)
                    if n not in remove_fields]
            col_names = [col_names[i] for i in keep]
            samps = samps[:, keep]

        # Write header + rows (root only)
        print(*dims, *col_names, sep=args.sep)
        for ploc, samp in zip(pts, samps):
            print(*ploc, *samp, sep=args.sep)
