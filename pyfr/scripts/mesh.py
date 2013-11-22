#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from argparse import ArgumentParser, FileType

import numpy as np

from pyfr.readers import get_reader_by_name, get_reader_by_extn, BaseReader
from pyfr.util import all_subclasses


def process_convert(args):
    # Get a suitable mesh reader instance
    if args.type:
        reader = get_reader_by_name(args.type, args.inmesh)
    else:
        extn = os.path.splitext(args.inmesh.name)[1]
        reader = get_reader_by_extn(extn, args.inmesh)

    # Get the mesh in the PyFR format
    mesh = reader.to_pyfrm()

    # Save to disk
    np.savez(args.outmesh, **mesh)


def main():
    ap = ArgumentParser(prog='pyfr-mesh', description='Generates and '
                        'manipulates PyFR mesh files')

    sp = ap.add_subparsers(help='sub-command help')

    # Mesh format conversion
    ap_convert = sp.add_parser('convert', help='convert --help')
    ap_convert.add_argument('inmesh', type=FileType('r'),
                            help='Input mesh file')
    ap_convert.add_argument('outmesh', type=FileType('wb'),
                            help='Output PyFR mesh file')
    types = [cls.name for cls in all_subclasses(BaseReader)]
    ap_convert.add_argument('-t', dest='type', choices=types, required=False,
                            help='Input file type; this is usually inferred '
                            'from the extension of inmesh')
    ap_convert.set_defaults(process=process_convert)

    args = ap.parse_args()
    args.process(args)

if __name__ == '__main__':
    main()
