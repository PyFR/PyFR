#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from argparse import ArgumentParser

import numpy as np

from pyfr.backends.cuda import CudaBackend
from pyfr.inifile import Inifile
from pyfr.integrators import get_integrator
from pyfr.rank_allocator import get_rank_allocation

def main():
    # Directory configuration
    cfg_dir, cfg_fname = os.path.split(sys.argv[1])
    os.chdir(cfg_dir)

    # Load the config file
    cfg = Inifile.load(cfg_fname)

    # Create a backend
    backend = CudaBackend()

    # Open up the mesh
    mesh = np.load(cfg.getpath('mesh', 'file'))

    # Get the mapping from physical ranks to MPI ranks
    rallocs = get_rank_allocation(mesh, cfg)

    # Construct the time integrator
    integrator = get_integrator(backend, rallocs, mesh, cfg)

    # Execute!
    integrator.run()


if __name__ == '__main__':
    main()
