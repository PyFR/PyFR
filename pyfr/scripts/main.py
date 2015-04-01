#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
import itertools as it
import os

import mpi4py.rc
mpi4py.rc.initialize = False

import h5py
from mpmath import mp

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.progress_bar import ProgressBar
from pyfr.rank_allocator import get_rank_allocation
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import read_pyfr_data
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import BaseWriter, get_writer_by_name, get_writer_by_extn


@mp.workdps(60)
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
    ap_import.set_defaults(process=process_import)

    # Partition command
    ap_partition = sp.add_parser('partition', help='partition --help')
    ap_partition.add_argument('np', help='number of partitions or a colon '
                              'delimited list of weighs')
    ap_partition.add_argument('mesh', help='input mesh file')
    ap_partition.add_argument('solns', nargs='*',
                              help='input solution files')
    ap_partition.add_argument(dest='outd', help='output directory')
    partitioners = sorted(cls.name for cls in subclasses(BasePartitioner))
    ap_partition.add_argument('-p', dest='partitioner', choices=partitioners,
                              help='partitioner to use')
    ap_partition.add_argument('--popt', dest='popts', action='append',
                              default=[], metavar='key:value',
                              help='partitioner-specific option')
    ap_partition.set_defaults(process=process_partition)

    # Export command
    ap_export = sp.add_parser('export', help='export --help',
                              description= 'Converts .pyfr[ms] files for '
                              'visualisation in external software.')
    ap_export.add_argument('meshf', help='PyFR mesh file to be converted')
    ap_export.add_argument('solnf', help='PyFR solution file to be converted')
    ap_export.add_argument('outf', type=FileType('wb'), help='Output filename')
    types = [cls.name for cls in subclasses(BaseWriter)]
    ap_export.add_argument('-t', dest='type', choices=types, required=False,
                           help='Output file type; this is usually inferred '
                           'from the extension of outf')
    ap_export.add_argument('-d', '--divisor', type=int, default=0,
                           help='Sets the level to which high order elements '
                           'are divided along each edge. The total node count '
                           'produced by divisor is equivalent to that of '
                           'solution order, which is used as the default. '
                           'Note: the output is linear between nodes, so '
                           'increased resolution may be required.')
    ap_export.add_argument('-p', '--precision', choices=['single', 'double'],
                           default='single', help='Selects the precision of '
                           'floating point numbers written to the output file; '
                           'single is the default.')
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
    for p in [ap_run, ap_restart]:
        p.add_argument('--backend', '-b', default='cuda',
                       help='Backend to use')
        p.add_argument('--progress', '-p', action='store_true',
                       help='show a progress bar')

    # Time average command
    ap_tavg = sp.add_parser('time-avg', help='time-avg --help',
                            description='Computes the mean solution of a '
                            'time-wise series of pyfrs files.')
    ap_tavg.add_argument('outf', type=str, help='Output PyFR solution file '
                         'name.')
    ap_tavg.add_argument('infs', type=str, nargs='+', help='Input '
                         'PyFR solution files to be time-averaged.')
    ap_tavg.add_argument('-l', '--limits', nargs=2, type=float,
                         help='Exclude solution files (passed in the infs '
                         'argument) that lie outside minimum and maximum '
                         'solution time limits.')
    ap_tavg.set_defaults(process=process_tavg)

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
    mesh = reader.to_pyfrm()

    # Save to disk
    with h5py.File(args.outmesh, 'w') as f:
        for k, v in mesh.items():
            f[k] = v


def process_partition(args):
    # Ensure outd is a directory
    if not os.path.isdir(args.outd):
        raise ValueError('Invalid output directory')

    # Partition weights
    if ':' in args.np:
        pwts = [int(w) for w in args.np.split(':')]
    else:
        pwts = [1]*int(args.np)

    # Partitioner-specific options
    opts = dict(s.split(':', 1) for s in args.popts)

    # Create the partitioner
    if args.partitioner:
        part = get_partitioner(args.partitioner, pwts, opts)
    else:
        for name in sorted(cls.name for cls in subclasses(BasePartitioner)):
            try:
                part = get_partitioner(name, pwts)
                break
            except RuntimeError:
                pass
        else:
            raise RuntimeError('No partitioners available')

    # Partition the mesh
    mesh, part_soln_fn = part.partition(read_pyfr_data(args.mesh))

    # Prepare the solutions
    solnit = (part_soln_fn(read_pyfr_data(s)) for s in args.solns)

    # Output paths/files
    paths = it.chain([args.mesh], args.solns)
    files = it.chain([mesh], solnit)

    # Iterate over the output mesh/solutions
    for path, data in zip(paths, files):
        # Compute the output path
        path = os.path.join(args.outd, os.path.basename(path.rstrip('/')))

        # Save to disk
        with h5py.File(path, 'w') as f:
            for k, v in data.items():
                f[k] = v


def process_export(args):
    # Get writer instance by specified type or outf extension
    if args.type:
        writer = get_writer_by_name(args.type, args)
    else:
        extn = os.path.splitext(args.outf.name)[1]
        writer = get_writer_by_extn(extn, args)

    # Write the output file
    writer.write_out()


def _process_common(args, mesh, soln, cfg):
    # Prefork to allow us to exec processes after MPI is initialised
    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

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
        args, read_pyfr_data(args.mesh), None, Inifile.load(args.cfg)
    )


def process_restart(args):
    mesh = read_pyfr_data(args.mesh)
    soln = read_pyfr_data(args.soln)

    # Ensure the solution is from the mesh we are using
    if soln['mesh_uuid'] != mesh['mesh_uuid']:
        raise RuntimeError('Invalid solution for mesh.')

    # Process the config file
    if args.cfg:
        cfg = Inifile.load(args.cfg)
    else:
        cfg = Inifile(soln['config'].item().decode())

    _process_common(args, mesh, soln, cfg)


def process_tavg(args):
    infs = {}

    # Interrogate files passed by the shell
    for fname in args.infs:
        # Load solution files and obtain solution times
        inf = read_pyfr_data(fname)
        cfg = Inifile(inf['stats'].item().decode())
        tinf = cfg.getfloat('solver-time-integrator', 'tcurr')

        # Retain if solution time is within limits
        if args.limits is None or args.limits[0] <= tinf <= args.limits[1]:
            infs[tinf] = inf

            # Verify that solutions were computed on the same mesh
            if inf['mesh_uuid'] != infs[list(infs.keys())[0]]['mesh_uuid']:
                raise RuntimeError('Solution files in scope were not computed '
                                   'on the same mesh')

    # Sort the solution times, check for sufficient files in scope
    stimes = sorted(infs.keys())
    if len(infs) <= 1:
        raise RuntimeError('More than one solution file is required to '
                           'compute an average')

    # Initialise progress bar, and the average with first solution
    pb = ProgressBar(0, 0, len(stimes), 0)
    avgs = {name: infs[stimes[0]][name].copy() for name in infs[stimes[0]]}
    solnfs = [name for name in avgs.keys() if name.startswith('soln')]

    # Weight the initialised trapezoidal mean
    dtnext = stimes[1] - stimes[0]
    for name in solnfs:
        avgs[name] *= 0.5*dtnext
    pb.advance_to(1)

    # Compute the trapezoidal mean up to the last solution file
    for i in range(len(stimes[2:])):
        dtlast = dtnext
        dtnext = stimes[i+2] - stimes[i+1]

        # Weight the current solution, then add to the mean
        for name in solnfs:
            avgs[name] += 0.5*(dtlast + dtnext)*infs[stimes[i+1]][name]
        pb.advance_to(i+2)

    # Weight final solution, update mean and normalise for elapsed time
    for name in solnfs:
        avgs[name] += 0.5*dtnext*infs[stimes[-1]][name]
        avgs[name] *= 1.0/(stimes[-1] - stimes[0])
    pb.advance_to(i+3)

    # Compute and assign stats for a time-averaged solution
    stats = Inifile()
    stats.set('time-average', 'tmin', stimes[0])
    stats.set('time-average', 'tmax', stimes[-1])
    stats.set('time-average', 'ntlevels', len(stimes))
    avgs['stats'] = stats.tostr()

    # Save to disk
    with h5py.File(args.outf, 'w') as f:
        for k, v in avgs.items():
            f[k] = v


if __name__ == '__main__':
    main()
