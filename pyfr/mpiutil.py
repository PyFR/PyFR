# -*- coding: utf-8 -*-

import os

from mpi4py import MPI

def get_comm_rank_root():
    comm = MPI.COMM_WORLD
    return comm, comm.rank, 0

def get_local_rank():
    envs = ['OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK']

    for ev in envs:
        if ev in os.environ:
            return int(os.environ[ev])
    else:
        raise RuntimeError('Unknown/unsupported MPI implementation')
