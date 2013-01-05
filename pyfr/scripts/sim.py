#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType

import numpy as np
from mpmath import mp

from pyfr import __version__ as version
from pyfr.backends.cuda import CudaBackend
from pyfr.inifile import Inifile
from pyfr.integrators import get_integrator
from pyfr.rank_allocator import get_rank_allocation
from pyfr.progress_bar import ProgressBar
from pyfr.util import get_comm_rank_root

def process_run(args):
    return np.load(args.mesh), None, Inifile.load(args.cfg)

def process_restart(args):
    mesh = np.load(args.mesh)
    soln = np.load(args.soln)

    # Ensure the solution is from the mesh we are using
    if soln['mesh_uuid'] != mesh['mesh_uuid']:
        raise RuntimeError('Invalid solution for mesh.')

    # Process the config file
    if args.cfg:
        cfg = Inifile.load(args.cfg)
    else:
        cfg = Inifile(soln['config'].item())

    return mesh, soln, cfg

@mp.workdps(60)
def main():
    ap = ArgumentParser(prog='pyfr-sim', description='Runs a PyFR simulation')
    ap.add_argument('--verbose', '-v', action='count')
    ap.add_argument('--progress', '-p', action='store_true',
                    help='show a progress bar')

    sp = ap.add_subparsers(help='sub-command help')

    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', type=FileType('rb'), help='mesh file')
    ap_run.add_argument('cfg', type=FileType('r'), help='config file')
    ap_run.set_defaults(process=process_run)

    ap_restart = sp.add_parser('restart', help='restart --help')
    ap_restart.add_argument('mesh', type=FileType('rb'), help='mesh file')
    ap_restart.add_argument('soln', type=FileType('rb'), help='solution file')
    ap_restart.add_argument('cfg', nargs='?', type=FileType('r'),
                            help='new config file')
    ap_restart.set_defaults(process=process_restart)

    # Parse the arguments
    args = ap.parse_args()
    mesh, soln, cfg = args.process(args)

    # Create a backend
    backend = CudaBackend(cfg)

    # Get the mapping from physical ranks to MPI ranks
    rallocs = get_rank_allocation(mesh, cfg)

    # Construct the time integrator
    integrator = get_integrator(backend, rallocs, mesh, soln, cfg)

    # If we are running interactively then create a progress bar
    if args.progress and get_comm_rank_root()[1] == 0:
        pb = ProgressBar(integrator.tstart, integrator.tcurr, integrator.tend)

        # Register a callback to update the bar after each step
        callb = lambda intg: pb.advance_to(intg.tcurr)
        integrator.completed_step_handlers.append(callb)

    # Execute!
    integrator.run()

if __name__ == '__main__':
    main()
