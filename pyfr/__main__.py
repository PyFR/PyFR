#!/usr/bin/env python
from argparse import ArgumentParser, FileType
import itertools as it
import os

import mpi4py.rc
mpi4py.rc.initialize = False

from pyfr._version import __version__
from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.plugins import BaseCLIPlugin
from pyfr.progress import ProgressBar, ProgressSequenceAction
from pyfr.rank_allocator import get_rank_allocation
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import (BaseWriter, get_writer_by_extn, get_writer_by_name,
                          write_pyfrms)


def main():
    ap = ArgumentParser(prog='pyfr')
    sp = ap.add_subparsers(help='sub-command help')

    # Common options
    ap.add_argument('--verbose', '-v', action='count')
    ap.add_argument('--version', '-V', action='version',
                    version=f'%(prog)s {__version__}')
    ap.add_argument('--progress', '-p', action=ProgressSequenceAction,
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

    # Partition command
    ap_partition = sp.add_parser('partition', help='partition --help')
    ap_partition.add_argument('np', help='number of partitions or a colon '
                              'delimited list of weights')
    ap_partition.add_argument('mesh', help='input mesh file')
    ap_partition.add_argument('solns', metavar='soln', nargs='*',
                              help='input solution files')
    ap_partition.add_argument('outd', help='output directory')
    partitioners = sorted(cls.name for cls in subclasses(BasePartitioner))
    ap_partition.add_argument('-p', dest='partitioner', choices=partitioners,
                              help='partitioner to use')
    ap_partition.add_argument('-r', dest='rnumf', type=FileType('w'),
                              help='output renumbering file')
    ap_partition.add_argument('-e', dest='elewts', action='append',
                              default=[], metavar='shape:weight',
                              help='element weighting factor or "balanced"')
    ap_partition.add_argument('--popt', dest='popts', action='append',
                              default=[], metavar='key:value',
                              help='partitioner-specific option')
    ap_partition.set_defaults(process=process_partition)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export.add_argument('meshf', help='PyFR mesh file to be converted')
    ap_export.add_argument('solnf', help='PyFR solution file to be converted')
    ap_export.add_argument('outf', type=str, help='output file')
    types = [cls.name for cls in subclasses(BaseWriter)]
    ap_export.add_argument('-t', dest='type', choices=types, required=False,
                           help='output file type; this is usually inferred '
                           'from the extension of outf')
    ap_export.add_argument('-f', '--field', dest='fields', action='append',
                           metavar='FIELD', required=False, help='what fields '
                           'should be output; may be repeated, by default all '
                           'fields are output')
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
        p.add_argument('--backend', '-b', choices=backends, required=True,
                       help='backend to use')

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
        extn = os.path.splitext(args.inmesh.name)[1]
        reader = get_reader_by_extn(extn, args.inmesh, args.progress)

    # Get the mesh in the PyFR format
    mesh = reader.to_pyfrm(args.lintol)

    # Save to disk
    with args.progress.start('Write mesh'):
        write_pyfrms(args.outmesh, mesh)


def process_partition(args):
    # Ensure outd is a directory
    if not os.path.isdir(args.outd):
        raise ValueError('Invalid output directory')

    # Read the mesh and query the partition info
    mesh = NativeReader(args.mesh)
    pinfo = mesh.partition_info('spt')

    # Partition weights
    if ':' in args.np:
        pwts = [int(w) for w in args.np.split(':')]
    else:
        pwts = [1]*int(args.np)

    # Element weights
    if args.elewts == ['balanced']:
        ewts = None
    elif len(pinfo) == 1:
        ewts = {next(iter(pinfo)): 1}
    else:
        ewts = {e: int(w) for e, w in (ew.split(':') for ew in args.elewts)}

    # Ensure all weights have been provided
    if ewts is not None and len(ewts) != len(pinfo):
        missing = ', '.join(set(pinfo) - set(ewts))
        raise ValueError(f'Missing element weights for: {missing}')

    # Partitioner-specific options
    opts = dict(s.split(':', 1) for s in args.popts)

    # Create the partitioner
    if args.partitioner:
        part = get_partitioner(args.partitioner, pwts, ewts, opts=opts)
    else:
        for name in sorted(cls.name for cls in subclasses(BasePartitioner)):
            try:
                part = get_partitioner(name, pwts, ewts)
                break
            except OSError:
                pass
        else:
            raise RuntimeError('No partitioners available')

    # Partition the mesh
    mesh, rnum, part_soln_fn = part.partition(mesh, args.progress)

    # Write the repartitioned mesh file
    with args.progress.start('Write mesh'):
        write_pyfrms(os.path.join(args.outd, os.path.basename(args.mesh)),
                     mesh)

    # Write out the renumbering table
    if args.rnumf:
        with args.progress.start('Write renumbering table'):
            print('etype,pold,iold,pnew,inew', file=args.rnumf)

            for etype, emap in sorted(rnum.items()):
                for k, v in sorted(emap.items()):
                    print(etype, *k, *v, sep=',', file=args.rnumf)

    # Repartition any solutions
    if args.solns:
        with args.progress.start_with_bar('Repartition solutions') as pbar:
            for ipath in pbar.start_with_iter(args.solns):
                # Compute the output path
                opath = os.path.join(args.outd, os.path.basename(ipath))

                # Save to disk
                write_pyfrms(opath, part_soln_fn(NativeReader(ipath)))


def process_export(args):
    # Get writer instance by specified type or outf extension
    if args.type:
        writer = get_writer_by_name(args.type, args)
    else:
        extn = os.path.splitext(args.outf)[1]
        writer = get_writer_by_extn(extn, args)

    # Write the output file
    with args.progress.start_with_bar('Write output') as pbar:
        writer.write_out(pbar)


def _process_common(args, mesh, soln, cfg):
    # Prefork to allow us to exec processes after MPI is initialised
    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

    # Work around issues with UCX-derived MPI libraries
    os.environ['UCX_MEMTYPE_CACHE'] = 'n'

    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    MPI.Init()

    # Ensure MPI is suitably cleaned up
    register_finalize_handler()

    # Create a backend
    backend = get_backend(args.backend, cfg)

    # Get the mapping from physical ranks to MPI ranks
    rallocs = get_rank_allocation(mesh, cfg)

    # Construct the solver
    solver = get_solver(backend, rallocs, mesh, soln, cfg)

    # If we are running interactively then create a progress bar
    if args.progress and MPI.COMM_WORLD.rank == 0:
        pbar = ProgressBar()
        pbar.start(solver.tend, start=solver.tstart, curr=solver.tcurr)

        # Register a callback to update the bar after each step
        solver.plugins.append(lambda intg: pbar(intg.tcurr))

    # Execute!
    solver.run()


def process_run(args):
    _process_common(
        args, NativeReader(args.mesh), None, Inifile.load(args.cfg)
    )


def process_restart(args):
    mesh = NativeReader(args.mesh)
    soln = NativeReader(args.soln)

    # Ensure the solution is from the mesh we are using
    if soln['mesh_uuid'] != mesh['mesh_uuid']:
        raise RuntimeError('Invalid solution for mesh.')

    # Process the config file
    if args.cfg:
        cfg = Inifile.load(args.cfg)
    else:
        cfg = Inifile(soln['config'])

    _process_common(args, mesh, soln, cfg)


if __name__ == '__main__':
    main()
