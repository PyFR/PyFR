# -*- coding: utf-8 -*-

from argparse import FileType
import os

from mpmath import mp

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler
from pyfr.rank_allocator import get_rank_allocation
from pyfr.progress_bar import ProgressBar
from pyfr.readers.native import read_pyfr_data
from pyfr.solvers import get_solver


@mp.workdps(60)
def _process_common(args, mesh, soln, cfg):
    # Prefork to allow us to exec processes after MPI is initialised
    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

    # Import and hence initialise MPI
    from mpi4py import MPI

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


def add_args(ap):
    ap.add_argument('--verbose', '-v', action='count')
    ap.add_argument('--backend', '-b', default='cuda', help='Backend to use')
    ap.add_argument('--progress', '-p', action='store_true',
                    help='show a progress bar')

    sp = ap.add_subparsers(help='sub-command help')

    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', help='mesh file')
    ap_run.add_argument('cfg', type=FileType('r'), help='config file')
    ap_run.set_defaults(process=process_run)

    ap_restart = sp.add_parser('restart', help='restart --help')
    ap_restart.add_argument('mesh', help='mesh file')
    ap_restart.add_argument('soln', help='solution file')
    ap_restart.add_argument('cfg', nargs='?', type=FileType('r'),
                            help='new config file')
    ap_restart.set_defaults(process=process_restart)


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
