import atexit
import ctypes
import math
import os
import sys
import weakref

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
            isinstance(exc, (KeyboardInterrupt, SystemExit))):
            import gc
            gc.collect()

            MPI.Finalize()
        # Otherwise forcefully abort
        else:
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)

    # Register our exit handler
    atexit.register(onexit)


def autofree(obj):
    def callfree(fromhandle, handle):
        fromhandle(handle).free()

    weakref.finalize(obj, callfree, obj.fromhandle, obj.handle)
    return obj


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

        return autofree(MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)).rank


def scal_coll(colfn, v, *args, **kwargs):
    dtype = int if isinstance(v, (int, np.integer)) else float
    v = np.array([v], dtype=dtype)
    colfn(mpi.IN_PLACE, v, *args, **kwargs)
    return dtype(v[0])


def get_start_end_csize(comm, n):
    rank, size = comm.rank, comm.size

    # Determine how much data each rank is responsible for
    csize = max(-(-n // size), 1)

    # Determine which part of the dataset we should handle
    return min(rank*csize, n), min((rank + 1)*csize, n), csize


class AlltoallMixin:
    @staticmethod
    def _count_to_disp(count):
        return np.concatenate(([0], np.cumsum(count[:-1])))

    @staticmethod
    def _disp_to_count(disp, n):
        return np.concatenate((disp[1:] - disp[:-1], [n - disp[-1]]))

    def _alltoallv(self, comm, sbuf, rbuf):
        svals = sbuf[0]

        # If we are dealing with scalar data then call Alltoallv directly
        if svals.dtype.names is None and svals.ndim == 1:
            comm.Alltoallv(sbuf, rbuf)
        # Else, we need to create a suitable derived datatype
        else:
            from mpi4py.util.dtlib import from_numpy_dtype

            dtype = svals.dtype

            if svals.ndim > 1:
                dtype = [('', dtype, svals.shape[1:])]

            dtype = autofree(from_numpy_dtype(dtype).Commit())
            comm.Alltoallv((*sbuf, dtype), (*rbuf, dtype))

    def _alltoallcv(self, comm, svals, scount, sdisps=None):
        # Exchange counts
        rcount = np.empty_like(scount)
        comm.Alltoall(scount, rcount)

        # Compute displacements
        rdisps = self._count_to_disp(rcount)
        sdisps = self._count_to_disp(scount) if sdisps is None else sdisps

        # Exchange values
        rvals = np.empty((rcount.sum(), *svals.shape[1:]), dtype=svals.dtype)
        rbuf = (rvals, (rcount, rdisps))
        self._alltoallv(comm, (svals, (scount, sdisps)), rbuf)

        return rbuf


class BaseGathererScatterer(AlltoallMixin):
    def __init__(self, comm, aidx):
        self.comm = comm

        # Determine array size
        n = aidx[-1] if len(aidx) else -1
        n = scal_coll(comm.Allreduce, n, op=mpi.MAX) + 1

        # Determine which part of the dataset we should handle
        self.start, self.end, csize = get_start_end_csize(comm, n)

        # Map each index to its associated rank
        adisps = np.searchsorted(aidx, csize*np.arange(comm.size))
        acount = np.diff(adisps, append=len(aidx))

        # Exchange the indices
        bidx, (bcount, bdisps) = self._alltoallcv(comm, aidx, acount, adisps)

        # Save the count and displacement information
        self.acountdisps = (acount, adisps)
        self.bcountdisps = (bcount, bdisps)

        # Return the index information
        return bidx


class Scatterer(BaseGathererScatterer):
    def __init__(self, comm, idx):
        idx = np.asanyarray(idx, dtype=int)

        # Eliminate duplicates from our index array
        ridx, self.rinv = np.unique(idx, return_inverse=True)

        self.sidx = super().__init__(comm, ridx) - self.start

        # Save the receive count
        self.cnt = len(ridx)

    def __call__(self, dset, didxs=(...,)):
        rcount, rdisps = self.acountdisps
        scount, sdisps = self.bcountdisps

        # Read the data
        svals = dset[self.start:self.end, *didxs][self.sidx]

        # Allocate space for receiving the data
        rvals = np.empty((self.cnt, *svals.shape[1:]), dtype=svals.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, (scount, sdisps)),
                        (rvals, (rcount, rdisps)))

        # Unpack the data
        return rvals[self.rinv]


class Gatherer(BaseGathererScatterer):
    def __init__(self, comm, idx):
        idx = np.asanyarray(idx, dtype=int)

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
        self._alltoallv(self.comm, (svals, (scount, sdisps)),
                        (rvals, (rcount, rdisps)))

        # Sort our received data
        return rvals[self.rinv]


class SparseScatterer(AlltoallMixin):
    def __init__(self, comm, iset, aidx):
        self.comm = comm

        # Sort our indices
        ainv = np.argsort(aidx)
        bidx = aidx[ainv]

        # Determine the array size
        n = len(iset)

        # Determine which part of the dataset we should handle
        self.start, self.end, _ = get_start_end_csize(comm, n)

        # Read our portion of the sorted index table
        cidx = iset[self.start:self.end]

        # Tell other ranks what region we have
        region = np.array([cidx.min(initial=n), cidx.max(initial=n) + 1])
        minmax = np.empty(2*comm.size, dtype=int)
        comm.Allgather(region, minmax)

        # Determine which rank, if any, has each of our desired indices
        didx = np.split(bidx, np.searchsorted(bidx, minmax))[1::2]
        dcount = np.array([len(s) for s in didx])

        # Exchange indices
        eidx, (ecount, edisps) = self._alltoallcv(comm, np.concatenate(didx),
                                                  dcount)

        # See which of these indices are present
        mask = np.isin(eidx, cidx, assume_unique=True)
        sidx = eidx[mask]
        scount = np.array([m.sum() for m in np.split(mask, edisps[1:])])
        sdisps = self._count_to_disp(scount)

        # Make a note of which indices we have
        self.sidx = np.searchsorted(cidx, sidx)
        self.scountdisps = (scount, sdisps)

        # Exchange the present indices
        ridx, self.rcountdisps = self._alltoallcv(comm, sidx, scount,
                                                  sdisps)

        self.ridx = ainv[np.searchsorted(bidx, ridx)]
        self.cnt = self.rcountdisps[0].sum()

    def __call__(self, dset, didxs=(...,)):
        scount, sdisps = self.scountdisps
        rcount, rdisps = self.rcountdisps

        # Read and appropriately reorder our send data
        svals = dset[self.start:self.end, *didxs][self.sidx]

        # Allocate space for receiving the data
        rvals = np.empty((self.cnt, *svals.shape[1:]), dtype=svals.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, (scount, sdisps)),
                        (rvals, (rcount, rdisps)))

        return rvals


class Sorter(AlltoallMixin):
    typemap = {
        np.int8: np.uint8, np.int16: np.uint16,
        np.int32: np.uint32, np.int64: np.uint64,
        np.float32: np.int32, np.float64: np.int64
    }

    def __init__(self, comm, keys):
        self.comm = comm

        # Locally sort our outbound keys
        self.sidx = np.argsort(keys)
        skeys = keys[self.sidx]

        # Determine the total size of the array
        size = scal_coll(comm.Allreduce, len(keys))

        start, end, csize = get_start_end_csize(comm, size)
        self.cnt = end - start

        # Determine what to send to each rank
        sdisps = self._splitters(skeys, start)
        scount = self._disp_to_count(sdisps, len(keys))
        self.scountdisps = (scount, sdisps)

        # Exchange the keys
        rkeys, self.rcountdisps = self._alltoallcv(comm, skeys, scount,
                                                   sdisps)

        # Locally sort our inbound keys
        self.ridx = np.argsort(rkeys)
        self.keys = rkeys[self.ridx]

    def _transform_keys(self, skeys):
        dtype = skeys.dtype

        if np.issubdtype(dtype, np.unsignedinteger):
            return skeys
        elif np.issubdtype(dtype, np.signedinteger):
            udtype = self.typemap[dtype]
            return skeys.view(udtype) ^ udtype(np.iinfo(dtype).max + 1)
        elif np.issubdtype(dtype, np.floating):
            shift = 8*dtype.itemsize - 1
            idtype = self.typemap[dtype]
            udtype = self.typemap[idtype]

            mask = (skeys.view(idtype) >> shift).view(udtype)
            mask |= udtype(1 << shift)
            return skeys.view(udtype) ^ mask
        else:
            raise ValueError('Unsupported dtype')

    def _splitters(self, skeys, r):
        # Transform the keys so they're unsigned integers
        skeys = self._transform_keys(skeys)

        # Determine the minimum and maximum values in the array
        kmin = self.comm.allreduce(int(skeys[0]), op=mpi.MIN)
        kmax = self.comm.allreduce(int(skeys[-1]), op=mpi.MAX)

        # Compute the number of bits in the key space
        W = math.ceil(math.log2(kmax - kmin + 1))

        e, rt = 0, 0
        q = np.empty(self.comm.size, dtype=skeys.dtype)

        for i in range(W - 1, -1, -1):
            # Compute and gather the probes
            q[self.comm.rank] = e + 2**i
            self.comm.Allgather(mpi.IN_PLACE, q)

            # Obtain the global location of each probe
            t = np.searchsorted(skeys, q)
            self.comm.Reduce_scatter_block(mpi.IN_PLACE, t)

            if t[0] <= r:
                e, rt = e + 2**i, t[0]

        q[self.comm.rank] = e
        self.comm.Allgather(mpi.IN_PLACE, q)

        # Count the occurances of each probe in skeys
        ubnd = np.searchsorted(skeys, q, side='right')
        lbnd = np.searchsorted(skeys, q, side='left')
        ld = ubnd - lbnd

        # Compute the global position of each probe
        gd = np.zeros_like(ld)
        self.comm.Exscan(ld, gd)

        q[self.comm.rank] = r - rt
        self.comm.Allgather(mpi.IN_PLACE, q)

        return lbnd + np.maximum(0, np.minimum(ld, q.astype(int) - gd))

    def __call__(self, svals):
        # Locally sort our data
        svals = svals[self.sidx]

        # Allocate space for receiving the data
        rvals = np.empty((self.cnt, *svals.shape[1:]), dtype=svals.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, self.scountdisps),
                        (rvals, self.rcountdisps))

        # Locally sort our received data
        return rvals[self.ridx]


class _MPI_Funcs:
    def __init__(self):
        from mpi4py import MPI

        self._lib = ctypes.CDLL(MPI.__file__)

    def __getattr__(self, attr):
        func = getattr(self._lib, f'MPI_{attr}')
        return ctypes.cast(func, ctypes.c_void_p).value


class _MPI:
    def __init__(self):
        self.funcs = _MPI_Funcs()

    def addrof(self, obj):
        from mpi4py import MPI

        return MPI._addressof(obj)

    def __getattr__(self, attr):
        from mpi4py import MPI

        return getattr(MPI, attr)


mpi = _MPI()
