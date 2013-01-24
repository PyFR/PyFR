#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from tempfile import NamedTemporaryFile
from argparse import ArgumentParser, FileType

import numpy as np

from pyfr.util import rm

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

def main():
    ap = ArgumentParser(prog='pyfr-postp', description='Post processes a '
                        'PyFR simulation')

    sp = ap.add_subparsers(help='sub-command help')

    ap_merge = sp.add_parser('pack', help='pack --help', description='Packs a '
                             'pyfrs-directory into a pyfrs-file.  If no '
                             'output file is specified then that of the '
                             'input directory is taken.  This command will '
                             'replace any existing file or directory.')
    ap_merge.add_argument('indir', metavar='in',
                          help='Input PyFR solution directory')
    ap_merge.add_argument('outf', metavar='out', nargs='?',
                          help='Out PyFR solution file')
    ap_merge.set_defaults(process=process_pack)

    # Parse the arguments
    args = ap.parse_args()
    args.process(args)

if __name__ == '__main__':
    main()
