# -*- coding: utf-8 -*-

import atexit
import os
import sys


def register_finalize_handler():
    import mpi4py.rc
    from mpi4py import MPI

    # Prevent mpi4py from calling MPI_Finalize
    mpi4py.rc.finalize = False

    # Intercept any uncaught exceptions
    class ExceptHook(object):
        def __init__(self):
            self.exception = None

            self._orig_excepthook = sys.excepthook
            sys.excepthook = self._excepthook

        def _excepthook(self, exc_type, exc, *args):
            self.exception = exc
            self._orig_excepthook(exc_type, exc, *args)

    # Register our exception hook
    excepthook = ExceptHook()

    def onexit():
        if not MPI.Is_initialized() or MPI.Is_finalized():
            return

        # Get the current exception (if any)
        exc = excepthook.exception

        # If we are exiting normally then call MPI_Finalize
        if (MPI.COMM_WORLD.size == 1 or exc is None or
            isinstance(exc, KeyboardInterrupt) or
            (isinstance(exc, SystemExit) and exc.code == 0)):
            MPI.Finalize()
        # Otherwise forcefully abort
        else:
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)

    # Register our exit handler
    atexit.register(onexit)


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


def get_mpi(attr):
    from mpi4py import MPI

    return getattr(MPI, attr.upper())
