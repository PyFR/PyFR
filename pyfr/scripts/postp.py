# -*- coding: utf-8 -*-

from argparse import FileType
import os

import numpy as np

from pyfr.inifile import Inifile
from pyfr.progress_bar import ProgressBar
from pyfr.readers.native import read_pyfr_data
from pyfr.util import subclasses
from pyfr.writers import BaseWriter, get_writer_by_name, get_writer_by_extn


def add_args(ap):
    sp = ap.add_subparsers(help='sub-command help')

    # Convert command
    ap_conv = sp.add_parser('convert', help='convert --help', description=
                            'Converts .pyfr[ms] files for visualisation '
                            'in external software.')
    ap_conv.add_argument('meshf', help='PyFR mesh file to be converted')
    ap_conv.add_argument('solnf', help='PyFR solution file to be converted')
    ap_conv.add_argument('outf', type=FileType('wb'), help='Output filename')
    types = [cls.name for cls in subclasses(BaseWriter)]
    ap_conv.add_argument('-t', dest='type', choices=types, required=False,
                         help='Output file type; this is usually inferred '
                         'from the extension of outf')
    ap_conv.add_argument('-d', '--divisor', type=int, default=0,
                         help='Sets the level to which high order elements '
                         'are divided along each edge. The total node count '
                         'produced by divisor is equivalent to that of '
                         'solution order, which is used as the default. '
                         'Note: the output is linear between nodes, so '
                         'increased resolution may be required.')
    ap_conv.add_argument('-p', '--precision', choices=['single', 'double'],
                         default='single', help='Selects the precision of '
                         'floating point numbers written to the output file; '
                         'single is the default.')
    ap_conv.set_defaults(process=process_convert)

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


def process_convert(args):
    # Get writer instance by specified type or outf extension
    if args.type:
        writer = get_writer_by_name(args.type, args)
    else:
        extn = os.path.splitext(args.outf.name)[1]
        writer = get_writer_by_extn(extn, args)

    # Write the output file
    writer.write_out()

    args.outf.close()


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

    outf = open(args.outf, 'wb')
    np.savez(outf, **avgs)
