#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import pyfr.scripts.mesh
import pyfr.scripts.sim
import pyfr.scripts.postp


def main():
    ap = ArgumentParser(prog='pyfr')
    sp = ap.add_subparsers(dest='cmd', help='sub-command help')

    # Subparsers
    sap = {
        'mesh': sp.add_parser('mesh', help='mesh sub-command help'),
        'sim': sp.add_parser('sim', help='sim sub-command help'),
        'postp': sp.add_parser('postp', help='postp sub-command help')
    }

    # Populate the subcommands
    pyfr.scripts.mesh.add_args(sap['mesh'])
    pyfr.scripts.sim.add_args(sap['sim'])
    pyfr.scripts.postp.add_args(sap['postp'])

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        sap.get(args.cmd, ap).print_help()


if __name__ == '__main__':
    main()
