#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from argparse import ArgumentParser, FileType
from collections import OrderedDict
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np

from pyfr.inifile import Inifile
from pyfr.progress_bar import ProgressBar
from pyfr.readers.native import read_pyfr_data
from pyfr.util import rm, all_subclasses
from pyfr.writers import get_writer_by_name, get_writer_by_extn, BaseWriter


def process_pack(args):
    # List the contents of the directory
    relnames = os.listdir(args.indir)

    # Get the absolute file names and extension-less file names
    absnames = [os.path.join(args.indir, f) for f in relnames]
    repnames = [f[:-4] for f in relnames]

    # Open/load the files
    files = [np.load(f, mmap_mode='r') for f in absnames]

    # Get the output pyfrs file name
    outname = args.outf or args.indir.rstrip('/')

    # Determine the dir and prefix of the temp file
    dirname, basename = os.path.split(outname)

    # Create a named temp file
    tempf = NamedTemporaryFile(prefix=basename, dir=dirname, delete=False)

    try:
        # Write the contents of the directory out as an npz (pyfrs) file
        np.savez(tempf, **dict(zip(repnames, files)))
        tempf.close()

        # Remove the output path if it should exist
        if os.path.exists(outname):
            rm(outname)

        # Rename the temp file into place
        os.rename(tempf.name, outname)
    except:
        # Clean up the temporary file
        if os.path.exists(tempf.name):
            os.remove(tempf.name)

        # Re-raise
        raise


def process_unpack(args):
    # Load the file
    inf = np.load(args.inf)

    # Determine the dir and prefix of the input file
    dirname, basename = os.path.split(args.inf.name)

    # Get the output pyfrs directory name
    outdir = args.outdir or args.inf.name

    # Create a temporary directory
    tmpdir = mkdtemp(prefix=basename, dir=dirname)

    # Write out the files to this temporary directory
    try:
        for n, d in inf.iteritems():
            np.save(os.path.join(tmpdir, n), d)

        # Remove the output path if it should exist
        if os.path.exists(outdir):
            os.remove(outdir)

        # Rename the temporary directory into place
        os.rename(tmpdir, outdir)
    except:
        # Clean up the temporary directory
        if os.path.exists(tmpdir):
            rm(tmpdir)

        # Re-raise
        raise


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
        tinf = Inifile(inf['stats']).getfloat('solver-time-integrator',
                                              'tcurr')

        # Retain if solution time is within limits
        if args.limits is None or args.limits[0] <= tinf <= args.limits[1]:
            infs[tinf] = inf

            # Verify that solutions were computed on the same mesh
            if inf['mesh_uuid'] != infs[infs.keys()[0]]['mesh_uuid']:
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
    for i in xrange(len(stimes[2:])):
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


def main():
    ap = ArgumentParser(prog='pyfr-postp', description='Post processes a '
                        'PyFR simulation')

    sp = ap.add_subparsers(help='sub-command help')

    ap_pack = sp.add_parser('pack', help='pack --help', description='Packs a '
                            'pyfr[ms]-directory into a pyfr[ms]-file.  If no '
                            'output file is specified then that of the '
                            'input directory is taken.  This command will '
                            'replace any existing file or directory.')
    ap_pack.add_argument('indir', metavar='in',
                         help='Input PyFR mesh/solution directory')
    ap_pack.add_argument('outf', metavar='out', nargs='?',
                         help='Out PyFR mesh/solution file')
    ap_pack.set_defaults(process=process_pack)

    ap_unpack = sp.add_parser('unpack', help='unpack --help', description=
                              'Unpacks a pyfr[ms]-file into a '
                              'pyfr[ms]-directory. If no output directory '
                              'name is specified then a directory named '
                              'after the input file is taken. This command '
                              'will remove any existing output file or '
                              'directory.')
    ap_unpack.add_argument('inf', metavar='in', type=FileType('rb'),
                           help='Input PyFR mesh/solution file')
    ap_unpack.add_argument('outdir', metavar='out', nargs='?',
                           help='Out PyFR mesh/solution directory')
    ap_unpack.set_defaults(process=process_unpack)

    ap_conv = sp.add_parser('convert', help='convert --help', description=
                            'Converts .pyfr[ms] files for visualisation '
                            'in external software.')
    ap_conv.add_argument('meshf', help='PyFR mesh file to be converted')
    ap_conv.add_argument('solnf', help='PyFR solution file to be converted')
    ap_conv.add_argument('outf', type=FileType('wb'), help='Output filename')
    types = [cls.name for cls in all_subclasses(BaseWriter)]
    ap_conv.add_argument('-t', dest='type', choices=types, required=False,
                         help='Output file type; this is usually inferred '
                         'from the extension of outf')

    ap_pop = ap_conv.add_subparsers(help='Choose output mode for high-order '
                                    'data.')

    ap_pdiv = ap_pop.add_parser('divide', help='paraview *args divide --help',
                                description='Divides high-order elements '
                                'into multiple low-order elements.  All '
                                'elements are split to the same level.')

    ap_pdiv.add_argument('-d', '--divisor', type=int, choices=range(1, 17),
                         default=0, help='Sets the level to which high '
                         'order elements are divided along each edge. The '
                         'total node count produced by divisor is equivalent '
                         'to that of solution order, which is used as the '
                         'default. Note: the output is linear between '
                         'nodes, so increased resolution may be required.')
    ap_pdiv.add_argument('-p', '--precision', choices=['single', 'double'],
                         default='single', help='Selects the precision of '
                         'floating point numbers written to the output file; '
                         'single is the default.')
    ap_pdiv.set_defaults(process=process_convert)

    ap_pap = ap_pop.add_parser('append', help='paraview *args append --help',
                               description='High-order solutions are written '
                               'as high-order data appended to low-order '
                               'elements.  A Paraview plugin recursively '
                               'bisects each element to achieve a requirement '
                               'on the relative error of a solution variable. '
                               'Paraview requires the high-order plugin '
                               'written by Sébastien Blaise, available from: '
                               'http://perso.uclouvain.be/sebastien.blaise/'
                               'tools.html')
    ap_pap.set_defaults(divisor=None, precision='double',
                        process=process_convert)

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
    args.process(args)

if __name__ == '__main__':
    main()
