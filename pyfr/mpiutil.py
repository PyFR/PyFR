import atexit
import os
import sys

import numpy as np


def init_mpi():
    import mpi4py.rc
    from mpi4py import MPI

    # Prefork to allow us to exec processes after MPI is initialised
    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

    # Work around issues with UCX-derived MPI libraries
    os.environ['UCX_MEMTYPE_CACHE'] = 'n'

    # Manually initialise MPI
    MPI.Init()

    # Prevent mpi4py from calling MPI_Finalize
    mpi4py.rc.finalize = False

    # Intercept any uncaught exceptions
    class ExceptHook:
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
    envs = [
        'MV2_COMM_WORLD_LOCAL_RANK',
        'OMPI_COMM_WORLD_LOCAL_RANK',
        'SLURM_LOCALID'
    ]

    for ev in envs:
        if ev in os.environ:
            return int(os.environ[ev])
    else:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED).rank


def scal_coll(colfn, v, *args, **kwargs):
    dtype = int if isinstance(v, (int, np.integer)) else float
    v = np.array([v], dtype=dtype)
    colfn(mpi.IN_PLACE, v, *args, **kwargs)
    return dtype(v[0])


class BaseGathererScatterer:
    def __init__(self, comm, aidx):
        self.comm = comm
        rank, size = comm.rank, comm.size

        # Determine array size
        n = aidx[-1] if len(aidx) else -1
        n = scal_coll(comm.Allreduce, n, op=mpi.MAX) + 1

        # Determine how much data each rank is responsible for
        csize = max(-(-n // size), 1)

        # Determine which part of the dataset we should handle
        self.start = min(rank*csize, n)
        self.end = min((rank + 1)*csize, n)

        # Map each index to its associated rank
        adisps = np.searchsorted(aidx, csize*np.arange(size))
        acount = np.diff(adisps, append=len(aidx))

        # Exchange these counts
        bcount = np.empty_like(acount)
        comm.Alltoall(acount, bcount)

        # Determine the associated displacements
        bdisps = np.concatenate(([0], np.cumsum(bcount[:-1])))

        # With this exchange the indices
        bidx = np.empty(np.sum(bcount), dtype=int)
        comm.Alltoallv((aidx, (acount, adisps)), (bidx, (bcount, bdisps)))

        # Save the count and displacement information
        self.acountdisps = (acount, adisps)
        self.bcountdisps = (bcount, bdisps)

        # Return the index information
        return bidx

    def _alltoallv(self, sbuf, rbuf):
        svals = sbuf[0]

        # If we are dealing with scalar data then call Alltoallv directly
        if svals.dtype.names is None and svals.ndim == 1:
            self.comm.Alltoallv(sbuf, rbuf)
        # Else, we need to create a suitable derived datatype
        else:
            from mpi4py.util.dtlib import from_numpy_dtype

            dtype = svals.dtype

            if svals.ndim > 1:
                dtype = [('', dtype, svals.shape[1:])]

            dtype = from_numpy_dtype(dtype).Commit()

            try:
                self.comm.Alltoallv((*sbuf, dtype), (*rbuf, dtype))
            finally:
                dtype.Free()


class Scatterer(BaseGathererScatterer):
    def __init__(self, comm, idx):
        # Eliminate duplicates from our index array
        ridx, self.rinv = np.unique(idx, return_inverse=True)

        self.sidx = super().__init__(comm, ridx)

        # Save the receive count
        self.cnt = len(ridx)

    def __call__(self, dset):
        rcount, rdisps = self.acountdisps
        scount, sdisps = self.bcountdisps

        # Read the data
        svals = dset[self.start:self.end][self.sidx - self.start]

        # Allocate space for receiving the data
        rvals = np.empty((self.cnt, *svals.shape[1:]), dtype=svals.dtype)

        # Perform the exchange
        self._alltoallv((svals, (scount, sdisps)), (rvals, (rcount, rdisps)))

        # Unpack the data
        return rvals[self.rinv]


class Gatherer(BaseGathererScatterer):
    def __init__(self, comm, idx):
        idx = np.asanyarray(idx)

        # Sort our send array
        self.sinv = np.argsort(idx)
        self.sidx = idx[self.sinv]

        bidx = super().__init__(comm, self.sidx)

        # Determine how to sort the data we will receive
        self.rinv = np.argsort(bidx)
        self.ridx = bidx[self.rinv]

        # Note the source rank of each received element
        self.rsrc = np.repeat(np.arange(comm.size), self.bcountdisps[0])
        self.rsrc = self.rsrc[self.rinv].astype(np.int32)

        # Compute the total number of items and our offset
        self.cnt = cnt = len(self.ridx)
        self.tot = scal_coll(comm.Allreduce, cnt, op=mpi.SUM)
        self.off = scal_coll(comm.Exscan, cnt, op=mpi.SUM)
        self.off = self.off if comm.rank else 0

    def __call__(self, dset):
        scount, sdisps = self.acountdisps
        rcount, rdisps = self.bcountdisps

        # Sort the data we are going to be sending
        svals = np.ascontiguousarray(dset[self.sinv])

        # Allocate space for the data we will receive
        rvals = np.empty((self.cnt, *dset.shape[1:]), dtype=dset.dtype)

        # Perform the exchange
        self._alltoallv((svals, (scount, sdisps)), (rvals, (rcount, rdisps)))

        # Sort our received data
        return rvals[self.rinv]


class _MPI:
    def __getattr__(self, attr):
        from mpi4py import MPI

        return getattr(MPI, attr)


mpi = _MPI()
