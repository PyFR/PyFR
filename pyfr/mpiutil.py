# -*- coding: utf-8 -*-

import os

from mpi4py import MPI

from excepthook import excepthook

def init():
    MPI.Init_thread()
    MPI.COMM_WORLD.barrier()

def atexit():
    if not MPI.Is_initialized() or MPI.Is_finalized():
        return

    exc = excepthook.exception
    if MPI.COMM_WORLD.size > 1 and exc is not None and\
       not isinstance(exc, KeyboardInterrupt) and\
       (not isinstance(exc, SystemExit) or exc.code != 0):
        MPI.COMM_WORLD.Abort()
    else:
        MPI.Finalize()

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
