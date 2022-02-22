#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
import itertools as it
import os

import mpi4py.rc
mpi4py.rc.initialize = False

from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.progress_bar import ProgressBar
from pyfr.rank_allocator import get_rank_allocation
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import (BaseWriter, get_writer_by_name, get_writer_by_extn,
                          write_pyfrms)


def main():
    ap = ArgumentParser(prog='pyfr')
    sp = ap.add_subparsers(dest='cmd', help='sub-command help')

    # Common options
    ap.add_argument('--verbose', '-v', action='count')

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
                              help='element weighting factor')
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
        p.add_argument('--progress', '-p', action='store_true',
                       help='show a progress bar')

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
        reader = get_reader_by_name(args.type, args.inmesh)
    else:
        extn = os.path.splitext(args.inmesh.name)[1]
        reader = get_reader_by_extn(extn, args.inmesh)

    # Get the mesh in the PyFR format
    mesh = reader.to_pyfrm(args.lintol)

    # Save to disk
    write_pyfrms(args.outmesh, mesh)


def process_partition(args):
    # Ensure outd is a directory
    if not os.path.isdir(args.outd):
        raise ValueError('Invalid output directory')

    # Partition weights
    if ':' in args.np:
        pwts = [int(w) for w in args.np.split(':')]
    else:
        pwts = [1]*int(args.np)

    # Element weights
    if args.elewts:
        ewts = {e: int(w) for e, w in (ew.split(':') for ew in args.elewts)}
    else:
        ewts = {'quad': 6, 'tri': 3, 'tet': 3, 'hex': 18, 'pri': 10, 'pyr': 6}

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
    mesh, rnum, part_soln_fn = part.partition(NativeReader(args.mesh))

    # Prepare the solutions
    solnit = (part_soln_fn(NativeReader(s)) for s in args.solns)

    # Output paths/files
    paths = it.chain([args.mesh], args.solns)
    files = it.chain([mesh], solnit)

    # Iterate over the output mesh/solutions
    for path, data in zip(paths, files):
        # Compute the output path
        path = os.path.join(args.outd, os.path.basename(path.rstrip('/')))

        # Save to disk
        write_pyfrms(path, data)

    # Write out the renumbering table
    if args.rnumf:
        print('etype,pold,iold,pnew,inew', file=args.rnumf)

        for etype, emap in sorted(rnum.items()):
            for k, v in sorted(emap.items()):
                print(etype, *k, *v, sep=',', file=args.rnumf)


def process_export(args):
    # Get writer instance by specified type or outf extension
    if args.type:
        writer = get_writer_by_name(args.type, args)
    else:
        extn = os.path.splitext(args.outf)[1]
        writer = get_writer_by_extn(extn, args)

    # Write the output file
    writer.write_out()


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
        pb = ProgressBar(solver.tstart, solver.tcurr, solver.tend)

        # Register a callback to update the bar after each step
        callb = lambda intg: pb.advance_to(intg.tcurr)
        solver.completed_step_handlers.append(callb)

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
