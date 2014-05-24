# -*- coding: utf-8 -*-

import os
import sys


def register_excepthook():
    currhook = sys.excepthook

    def newhook(exctype, exc, *args):
        # See if mpi4py has been imported or not
        if 'mpi4py' in sys.modules:
            from mpi4py import MPI

            # If we are running with multiple ranks and are exiting
            # abnormally then abort
            if (MPI.COMM_WORLD.size > 1 and
                not isinstance(exc, KeyboardInterrupt) and
                (not isinstance(exc, SystemExit) or exc.code != 0)):
                MPI.COMM_WORLD.Abort(1)

        # Delegate to the previous handler
        currhook(exctype, exc, *args)

    # Install the new handler
    sys.excepthook = newhook


def get_comm_rank_root():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    return comm, comm.rank, 0


def get_local_rank():
    envs = ['OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK']

    for ev in envs:
        if ev in os.environ:
            return int(os.environ[ev])
    else:
        from mpi4py import MPI

        hostn = MPI.Get_processor_name()
        grank = MPI.COMM_WORLD.rank
        lrank = 0

        for i, n in enumerate(MPI.COMM_WORLD.allgather(hostn)):
            if i >= grank:
                break

            if hostn == n:
                lrank += 1

        return lrank
