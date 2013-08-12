#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from tempfile import NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, FileType

import numpy as np

from pyfr.inifile import Inifile
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
    # Build list of .pyfrs file names in range
    if args.rnge:
        try:
            # Find common part of .pyfrs file names & min/max of numbering
            prfx, fmin = args.infs[0].rsplit('.', 1)[0].rsplit('-', 1)
            prfx2, fmax = args.infs[1].rsplit('.', 1)[0].rsplit('-', 1)
        except:
            raise RuntimeError('Range expansion requires file naming of form:'
                               ' <constant_prefix>-<unique_number>.pyfrs')

        if prfx != prfx2:
            raise RuntimeError('Range expansion requires a constant prefix '
                               'in the file name')

        # Parse paths to external directories
        direc, prfx = os.path.split(prfx)
        if direc == '':
            direc = '.'

        # Find .pyfrs files within name range
        infs = []
        for filenm in os.listdir(direc):
            tmp = filenm.rsplit('.', 1)

            # Check for .pyfrs file names with same prefix as in args.infs
            if tmp[-1] == 'pyfrs' and tmp[0].rsplit('-', 1)[0] == prfx:
                fprfx, fsufx = tmp[0].rsplit('-', 1)

                # Select files within suffix range defined by args.infs
                if fsufx >= fmin and fsufx <= fmax:
                    infs.append(os.path.join(direc, filenm))

        args.infs = sorted(infs)

    if len(args.infs) <= 1:
        raise RuntimeError('More than one solution file is required for an '
                           'average')

    # Initialise the rolling average, load the time of first snapshot
    avg = read_pyfr_data(args.infs[0])
    tmin = Inifile(avg['stats'].item()).getfloat('time-integration', 'tcurr')
    cavg = {name: avg[name] for name in list(avg)}

    # Initialise the next file to be averaged, load the time and delta
    scur = read_pyfr_data(args.infs[1])
    tcur = Inifile(scur['stats'].item()).getfloat('time-integration', 'tcurr')
    dtcur = tcur - tmin

    # Verify that solutions were computed on the same mesh
    if scur['mesh_uuid'] != cavg['mesh_uuid']:
        raise RuntimeError('Solution files in scope were not computed on '
                           'the same mesh')

    # Weight the initialised trapezoidal rolling mean
    for name in avg.soln_files:
        cavg[name] *= 0.5*dtcur

    # Compute the rolling mean up to the last solution file
    for i, filenm in enumerate(args.infs[2:]):
        sys.stdout.write('\rProcessing file: %s' % args.infs[i])
        sys.stdout.flush()

        # Read in the next solution, obtain the next time and delta
        snxt = read_pyfr_data(filenm)
        tnxt = Inifile(snxt['stats'].item()).getfloat('time-integration',
                                                      'tcurr')
        dtnxt = tnxt - tcur

        # Verify that solutions were computed on the same mesh
        if snxt['mesh_uuid'] != cavg['mesh_uuid']:
            raise RuntimeError('Solution files in scope were not computed on '
                               'the same mesh')

        # Weight the current solution, then add to the rolling mean
        weight = 0.5*(dtcur + dtnxt)
        for name in avg.soln_files:
            cavg[name] += scur[name]*weight

        # Roll solution pointer, time level and delta
        scur = snxt
        tcur = tnxt
        dtcur = dtnxt

    # Weight final solution, update mean and normalise for elapsed time
    for name in avg.soln_files:
        cavg[name] += 0.5*dtcur*scur[name]
        cavg[name] *= 1.0/(tcur - tmin)

    # Compute and assign stats for a time-averaged solution
    stats = Inifile()
    stats.set('time-average', 'tmin', tmin)
    stats.set('time-average', 'tmax', tcur)
    stats.set('time-average', 'ntlevels', len(args.infs))
    cavg['stats'] = stats.tostr()

    # Assign output file name
    if args.outfile is None:
        if args.rnge:
            args.outfile = '-'.join([prfx, 'mean', fmin, 'to',
                                     fmax]) + '.pyfrs'
        else:
            args.outfile = 'time-averaged-solution.pyfrs'

    outf = open(args.outfile, 'wb')
    np.savez_compressed(outf, **cavg)


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
                            'series of PyFR solution (*.pyfrs) files')
    ap_tavg.add_argument('infs', type=str, nargs='+', help='Names '
                         'of the (*.pyfrs) solution files to be averaged')
    ap_tavg.add_argument('-o', '--outfile', type=str, help='Define '
                         'the name for the (*.pyfrs) output file')
    ap_tavg.add_argument('-r', '--range', action='store_true', dest='rnge',
                         help='Usage allows the list of input filenames '
                         'to be inferred from the contents of a directory.  '
                         'A "min" then "max" filename is passed for the '
                         'the [infs ...] arguments. Solution filenames must '
                         'be consistent and structured of the form:  '
                         '<constant-prefix>-<unique-numeric>.pyfrs, '
                         'where the <unique-numeric> is a non-negative '
                         'integer or real number')
    ap_tavg.set_defaults(process=process_tavg)

    # Parse the arguments
    args = ap.parse_args()
    args.process(args)

if __name__ == '__main__':
    main()
