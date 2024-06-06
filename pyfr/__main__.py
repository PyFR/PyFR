#!/usr/bin/env python
from argparse import ArgumentParser, FileType
from pathlib import Path
import re

import h5py
import mpi4py.rc
mpi4py.rc.initialize = False

from pyfr._version import __version__
from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.plugins import BaseCLIPlugin
from pyfr.progress import ProgressBar, ProgressSequenceAction
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import BaseWriter, get_writer_by_extn, get_writer_by_name


def main():
    ap = ArgumentParser(prog='pyfr')
    sp = ap.add_subparsers(help='sub-command help', metavar='command')

    # Common options
    ap.add_argument('-v', '--verbose', action='count',
                    help='increase verbosity')
    ap.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')
    ap.add_argument('-p', '--progress', action=ProgressSequenceAction,
                    help='show progress')

    # Import command
    ap_import = sp.add_parser('import', help='import --help')
    ap_import.add_argument('inmesh', type=FileType('r'),
                           help='input mesh file')
    ap_import.add_argument('outmesh', help='output PyFR mesh file')
    types = sorted(cls.name for cls in subclasses(BaseReader))
    ap_import.add_argument('-t', dest='type', choices=types,
                           help='input file type; this is usually inferred '
                           'from the extension of inmesh')
    ap_import.add_argument('-l', dest='lintol', type=float, default=1e-5,
                           help='linearisation tolerance')
    ap_import.set_defaults(process=process_import)

    # Partition subcommand
    ap_partition = sp.add_parser('partition', help='partition --help')
    ap_partition = ap_partition.add_subparsers()

    # List partitionings
    ap_partition_list = ap_partition.add_parser('list',
                                                help='partition list --help')
    ap_partition_list.add_argument('mesh', help='input mesh file')
    ap_partition_list.add_argument('-s', '--sep', default='\t',
                                   help='separator')
    ap_partition_list.set_defaults(process=process_partition_list)

    # Get info about a partitioning
    ap_partition_info = ap_partition.add_parser('info',
                                                help='partition info --help')
    ap_partition_info.add_argument('mesh', help='input mesh file')
    ap_partition_info.add_argument('name', help='partitioning name')
    ap_partition_info.add_argument('-s', '--sep', default='\t',
                                   help='separator')
    ap_partition_info.set_defaults(process=process_partition_info)

    # Add partitioning
    ap_partition_add = ap_partition.add_parser('add',
                                               help='partition add --help')
    ap_partition_add.add_argument('mesh', help='input mesh file')
    ap_partition_add.add_argument('np', help='number of partitions or a colon '
                                  'delimited list of weights')
    ap_partition_add.add_argument('name', nargs='?', help='partitioning name')
    ap_partition_add.add_argument('-f', '--force', action='count',
                                  help='overwrite existing partitioning')
    partitioners = sorted(cls.name for cls in subclasses(BasePartitioner))
    ap_partition_add.add_argument(
        '-p', dest='partitioner', choices=partitioners,
        help='partitioner to use'
    )
    ap_partition_add.add_argument(
        '-e', dest='elewts', action='append', default=[],
        metavar='shape:weight', help='element weighting factor or "balanced"'
    )
    ap_partition_add.add_argument(
        '--popt', dest='popts', action='append', default=[],
        metavar='key:value', help='partitioner-specific option'
    )
    ap_partition_add.set_defaults(process=process_partition_add)

    # Remove partitioning
    ap_partition_remove = ap_partition.add_parser(
        'remove', help='partition remove --help'
    )
    ap_partition_remove.add_argument('mesh', help='input mesh file')
    ap_partition_remove.add_argument('name', help='partitioning')
    ap_partition_remove.set_defaults(process=process_partition_remove)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export.add_argument('meshf', help='PyFR mesh file to be converted')
    ap_export.add_argument('solnf', help='PyFR solution file to be converted')
    ap_export.add_argument('outf', help='output file')
    types = [cls.name for cls in subclasses(BaseWriter)]
    ap_export.add_argument('-t', dest='type', choices=types, required=False,
                           help='output file type; this is usually inferred '
                           'from the extension of outf')
    ap_export.add_argument('-f', '--field', dest='fields', action='append',
                           metavar='FIELD', help='what fields should be '
                           'output; may be repeated, by default all fields '
                           'are output')
    output_options = ap_export.add_mutually_exclusive_group(required=False)
    output_options.add_argument('-d', '--divisor', type=int,
                                help='sets the level to which high order '
                                'elements are divided; output is linear '
                                'between nodes, so increased resolution '
                                'may be required')
    output_options.add_argument('-k', '--order', type=int, dest='order',
                                help='sets the order of high order elements')
    ap_export.add_argument('-p', '--precision', choices=['single', 'double'],
                           default='single', help='output number precision; '
                           'defaults to single')
    ap_export.add_argument('-b', '--boundary', dest='boundaries',
                           action='append', metavar='BOUNDARY',
                           help='boundary to output; may be repeated')
    ap_export.set_defaults(process=process_export)

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', help='mesh file')
    ap_run.add_argument('cfg', type=FileType('r'), help='config file')
    ap_run.set_defaults(process=process_run)

    # Restart command
    ap_restart = sp.add_parser('restart', help='restart --help')
    ap_restart.add_argument('mesh', help='mesh file')
    ap_restart.add_argument('soln', help='solution file')
    ap_restart.add_argument('cfg', nargs='?', type=FileType('r'),
                            help='new config file')
    ap_restart.set_defaults(process=process_restart)

    # Options common to run and restart
    backends = sorted(cls.name for cls in subclasses(BaseBackend))
    for p in [ap_run, ap_restart]:
        p.add_argument('-b', '--backend', choices=backends, required=True,
                       help='backend to use')
        p.add_argument('-p', '--pname', help='partitioning to use')

    # Plugin commands
    for scls in subclasses(BaseCLIPlugin, just_leaf=True):
        scls.add_cli(sp.add_parser(scls.name, help=f'{scls.name} --help'))

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()


def process_import(args):
    # Get a suitable mesh reader instance
    if args.type:
        reader = get_reader_by_name(args.type, args.inmesh, args.progress)
    else:
        extn = Path(args.inmesh.name).suffix
        reader = get_reader_by_extn(extn, args.inmesh, args.progress)

    # Write out the mesh
    reader.write(args.outmesh, args.lintol)


def process_partition_list(args):

    with h5py.File(args.mesh, 'r') as mesh:
        print('name', 'parts', sep=args.sep)

        for name, part in sorted(mesh['partitionings'].items()):
            nparts = len(part['eles'].attrs['regions'])
            print(name, nparts, sep=args.sep)


def process_partition_info(args):
    # Open the mesh
    with h5py.File(args.mesh, 'r') as mesh:
        # Read the partition region info from the mesh
        regions = mesh[f'partitionings/{args.name}/eles'].attrs['regions']

        # Print out the header
        print('part', *mesh['eles'], sep=args.sep)

        # Compute and output the number of elements in each partition
        for i, neles in enumerate(regions[:, 1:] - regions[:, :-1]):
            print(i, *neles, sep=args.sep)


def process_partition_add(args):
    # Open the mesh
    with h5py.File(args.mesh, 'r+') as mesh:
        # Determine the element types
        etypes = list(mesh['eles'])

        # Partition weights
        if ':' in args.np:
            pwts = [int(w) for w in args.np.split(':')]
        else:
            pwts = [1]*int(args.np)

        # Element weights
        if args.elewts == ['balanced']:
            ewts = None
        elif len(etypes) == 1:
            ewts = {etypes[0]: 1}
        else:
            ewts = {e: int(w) for e, w in (ew.split(':') for ew in args.elewts)}

        # Ensure all weights have been provided
        if ewts is not None and len(ewts) != len(etypes):
            missing = ', '.join(set(etypes) - set(ewts))
            raise ValueError(f'Missing element weights for: {missing}')

        # Get the partitioning name
        pname = args.name or str(len(pwts))
        if not re.match(r'\w+$', pname):
            raise ValueError('Invalid partitioning name')

        # Check it does not already exist unless --force is given
        if pname in mesh['partitionings'] and not args.force:
            raise ValueError('Partitioning already exists; use -f to replace')

        # Path to store the partitioning in the mesh
        ppath = f'partitionings/{pname}'

        # Partitioner-specific options
        opts = dict(s.split(':', 1) for s in args.popts)

        # Create the partitioner
        if args.partitioner:
            part = get_partitioner(args.partitioner, pwts, ewts, opts=opts)
        else:
            parts = sorted(cls.name for cls in subclasses(BasePartitioner))
            for name in parts:
                try:
                    part = get_partitioner(name, pwts, ewts)
                    break
                except OSError:
                    pass
            else:
                raise RuntimeError('No partitioners available')

        # Partition the mesh
        pinfo = part.partition(mesh, args.progress)
        (partitioning, pregions), (neighbours, nregions) = pinfo

        # Write out the new partitioning
        with args.progress.start('Write partitioning'):
            if ppath in mesh:
                mesh[f'{ppath}/eles'][:] = partitioning
                del mesh[f'{ppath}/neighbours']
            else:
                mesh[f'{ppath}/eles'] = partitioning

            mesh[f'{ppath}/neighbours'] = neighbours
            mesh[f'{ppath}/eles'].attrs['regions'] = pregions
            mesh[f'{ppath}/neighbours'].attrs['regions'] = nregions


def process_partition_remove(args):
    with h5py.File(args.mesh, 'r+') as mesh:
        mparts = mesh['partitionings']

        if args.name not in mparts:
            raise ValueError(f'Partitioning {args.name} does not exist')

        del mparts[args.name]


def process_export(args):
    # Manually initialise MPI
    init_mpi()

    comm, rank, root = get_comm_rank_root()

    # Get writer instance by specified type or outf extension
    if args.type:
        writer = get_writer_by_name(args.type, args)
    else:
        extn = Path(args.outf).suffix
        writer = get_writer_by_extn(extn, args)

    # Write the output file
    if extn == '.vtu':
        writer.write_vtu(args.outf)
    else:
        writer.write_pvtu(args.outf)


def _process_common(args, soln, cfg):
    # Manually initialise MPI
    init_mpi()

    comm, rank, root = get_comm_rank_root()

    # Read the mesh
    reader = NativeReader(args.mesh, pname=args.pname)
    mesh = reader.mesh

    # Load a provided solution, if any
    if soln is not None:
        soln = reader.load_soln(soln)

    # If we do not have a config file then take it from the solution
    if cfg is None:
        cfg = soln['config']

    # Create a backend
    backend = get_backend(args.backend, cfg)

    # Construct the solver
    solver = get_solver(backend, mesh, soln, cfg)

    # If we are running interactively then create a progress bar
    if args.progress and rank == root:
        pbar = ProgressBar()
        pbar.start(solver.tend, start=solver.tstart, curr=solver.tcurr)

        # Register a callback to update the bar after each step
        solver.plugins.append(lambda intg: pbar(intg.tcurr))

    # Execute!
    solver.run()


def process_run(args):
    _process_common(args, None, Inifile.load(args.cfg))


def process_restart(args):
    cfg = Inifile.load(args.cfg) if args.cfg else None
    _process_common(args, args.soln, cfg)


if __name__ == '__main__':
    main()
