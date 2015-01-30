#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
import itertools as it
import os

import numpy as np

from pyfr.partitioners import BasePartitioner, get_partitioner_by_name
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import read_pyfr_data
from pyfr.util import subclasses


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
        part = get_partitioner_by_name(args.partitioner, pwts, opts)
    else:
        for name in sorted(cls.name for cls in subclasses(BasePartitioner)):
            try:
                part = get_partitioner_by_name(name, pwts)
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

        # Open and save
        with open(path, 'wb') as f:
            np.savez(f, **data)


def main():
    ap = ArgumentParser(prog='pyfr-mesh', description='Generates and '
                        'manipulates PyFR mesh files')

    sp = ap.add_subparsers(help='sub-command help')

    # Mesh format conversion
    ap_convert = sp.add_parser('convert', help='convert --help')
    ap_convert.add_argument('inmesh', type=FileType('r'),
                            help='input mesh file')
    ap_convert.add_argument('outmesh', type=FileType('wb'),
                            help='output PyFR mesh file')
    types = sorted(cls.name for cls in subclasses(BaseReader))
    ap_convert.add_argument('-t', dest='type', choices=types,
                            help='input file type; this is usually inferred '
                            'from the extension of inmesh')
    ap_convert.set_defaults(process=process_convert)

    # Mesh and solution partitioning
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
                              help='partitioner-specific option ')
    ap_partition.set_defaults(process=process_partition)

    args = ap.parse_args()
    args.process(args)


if __name__ == '__main__':
    main()
