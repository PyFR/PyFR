import atexit
import contextlib
import ctypes
import math
import os
import sys
import time
import weakref

import numpy as np

from pyfr.cache import memoize


def init_mpi():
    import mpi4py.rc
    from mpi4py import MPI

    # Manually initialise MPI with thread support
    MPI.Init_thread()

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


def get_node_comm(comm):
    return autofree(comm.Split_type(mpi.COMM_TYPE_SHARED))


def get_local_rank():
    envs = [
        'MPI_LOCALRANKID',
        'MV2_COMM_WORLD_LOCAL_RANK',
        'OMPI_COMM_WORLD_LOCAL_RANK',
        'SLURM_LOCALID'
    ]

    for ev in envs:
        if ev in os.environ:
            return int(os.environ[ev])
    else:
        from mpi4py import MPI

        return get_node_comm(MPI.COMM_WORLD).rank


def scal_coll(colfn, v, *args, **kwargs):
    dtype = int if isinstance(v, (int, np.integer)) else float
    v = np.array([v], dtype=dtype)
    colfn(mpi.IN_PLACE, v, *args, **kwargs)
    return dtype(v[0])


def home_rank(gidxs, size):
    h = np.uint64(2654435761)*np.asarray(gidxs).view(np.uint64)
    return (h % size).astype(np.int32)


def get_start_end_csize(comm, n):
    rank, size = comm.rank, comm.size

    # Determine how much data each rank is responsible for
    csize = max(-(-n // size), 1)

    # Determine which part of the dataset we should handle
    return min(rank*csize, n), min((rank + 1)*csize, n), csize


def shared_ndarrays(comm, shape, dtype):
    dtype = np.dtype(dtype)

    # Gather the segment shapes
    shapes = comm.allgather(shape)
    sizes = [dtype.itemsize*math.prod(s) for s in shapes]

    # Allocate our segment inside of a single shared window
    win = mpi.Win.Allocate_shared(sizes[comm.rank], dtype.itemsize, comm=comm)
    win = autofree(win)

    # Wrap each ranks segment with an ndarray view
    views = [np.frombuffer(win.Shared_query(i)[0], dtype=dtype).reshape(s)
             for i, s in enumerate(shapes)]

    return win, views


class SharedWorkQueue:
    def __init__(self, comm, n, bsize):
        self.comm = comm
        self.bsize = bsize

        # Gather the number of items each rank on our node has
        self.counts = comm.allgather(n)
        self.nblocks = [-(-c // bsize) for c in self.counts]

        # Give each rank an atomic block cursor in a shared window
        self.win = win = mpi.Win.Allocate_shared(8, 8, comm=comm)
        bufs = [win.Shared_query(i)[0] for i in range(comm.size)]
        self.cursors = [np.frombuffer(b, dtype=np.int64) for b in bufs]
        self.cursors[comm.rank][0] = 0

        # Buffers for the atomic fetch-and-add operations
        self._one = np.ones(1, dtype=np.int64)
        self._res = np.empty(1, dtype=np.int64)

        # Publish our zeroed cursor before any peer can claim from it
        comm.barrier()

        # Open a passive access epoch spanning the entire window
        win.Lock_all()

    def _claim_from(self, r):
        # Atomically advance the block cursor of rank r
        self.win.Fetch_and_op(self._one, self._res, r, 0, mpi.SUM)
        self.win.Flush(r)

        return int(self._res[0])

    def claim(self):
        rank = self.comm.rank

        # Our own blocks are always claimed first, without any bookkeeping
        if (b := self._claim_from(rank)) < self.nblocks[rank]:
            lo = b*self.bsize

            return rank, slice(lo, min(lo + self.bsize, self.counts[rank]))

        # With our own queue drained, estimate what our peers have left
        rem = [n - int(c[0]) for c, n in zip(self.cursors, self.nblocks)]

        # Steal from the busiest peer first; cursors only ever increase
        for r in np.argsort(rem)[::-1]:
            if (b := self._claim_from(r)) < self.nblocks[r]:
                lo = b*self.bsize

                return int(r), slice(lo, min(lo + self.bsize, self.counts[r]))
        else:
            return None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.win.Unlock_all()

        # Ensure all thieves are finished before freeing the window
        self.comm.barrier()
        self.win.Free()


@contextlib.contextmanager
def mpi_progress(comm, pbar, n):
    pcomm = autofree(comm.Dup())

    # Obtain the total number of items and start the progress bar
    pbar.start(scal_coll(pcomm.Allreduce, n))

    buf = np.zeros(1, dtype=int)

    if pcomm.rank == 0:
        ndone, npeer = 0, pcomm.size - 1

        def poll():
            nonlocal ndone, npeer

            # Process any pending messages from peers
            while pcomm.Iprobe():
                pcomm.Recv(buf)
                if buf[0] < 0:
                    npeer -= 1
                else:
                    ndone += int(buf[0])

        def tick(k):
            nonlocal ndone

            ndone += k
            poll()
            pbar(ndone)

        yield tick

        # Wait until all peers have finished
        while npeer:
            poll()
            pbar(ndone)

            if npeer:
                time.sleep(0.05)
    else:
        def tick(k):
            buf[0] = k
            pcomm.Send(buf, dest=0)

        yield tick

        # Signal completion to the root rank
        buf[0] = -1
        pcomm.Send(buf, dest=0)


class AlltoallMixin:
    @staticmethod
    def _count_to_disp(count):
        disp = np.empty(len(count), dtype=count.dtype)
        disp[0] = 0
        np.cumsum(count[:-1], out=disp[1:])
        return disp

    @staticmethod
    def _disp_to_count(disp, n):
        return np.diff(disp, append=n)

    @memoize
    def _get_mpi_dtype(self, np_dtype, shape):
        if np_dtype.names is None and not shape:
            return None

        from mpi4py.util.dtlib import from_numpy_dtype

        dtype = np_dtype
        if shape:
            dtype = [('', dtype, shape)]

        return autofree(from_numpy_dtype(dtype).Commit())

    def _alltoallv_bufs(self, sbuf, rbuf):
        svals = sbuf[0]
        dtype = self._get_mpi_dtype(svals.dtype, svals.shape[1:])

        if dtype is None:
            return sbuf, rbuf
        else:
            return (*sbuf, dtype), (*rbuf, dtype)

    def _alltoallv(self, comm, sbuf, rbuf):
        return comm.Alltoallv(*self._alltoallv_bufs(sbuf, rbuf))

    def _alltoallv_init(self, comm, sbuf, rbuf):
        return comm.Alltoallv_init(*self._alltoallv_bufs(sbuf, rbuf))

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


class DistributedDirectory(AlltoallMixin):
    def __init__(self, comm, keys):
        self.comm = comm

        keys = np.asarray(keys, dtype=int)

        # Send each key to its home rank
        home = home_rank(keys, comm.size)
        self.sidx = np.argsort(home)
        scount = np.bincount(home, minlength=comm.size)
        sdisps = self._count_to_disp(scount)
        self.scountdisps = (scount, sdisps)

        recv, self.rcountdisps = self._alltoallcv(comm, keys[self.sidx],
                                                  scount, sdisps)

        # Reconstruct source ranks from receive counts
        rcount, _ = self.rcountdisps
        ranks = np.repeat(np.arange(comm.size, dtype=np.int32), rcount)

        # Sort received keys for searchsorted in lookup
        self.ridx = np.argsort(recv, kind='stable')
        self.keys = recv[self.ridx]
        self.ranks = ranks[self.ridx]

        self.ridx_inv = np.empty_like(self.ridx)
        self.ridx_inv[self.ridx] = np.arange(len(self.ridx))

    def scatter(self, values):
        rv, _ = self._alltoallcv(self.comm, values[self.sidx],
                                 *self.scountdisps)
        return rv[self.ridx]

    def gather(self, values):
        rv, _ = self._alltoallcv(self.comm, values[self.ridx_inv],
                                 *self.rcountdisps)

        out = np.empty_like(rv)
        out[self.sidx] = rv
        return out

    def lookup(self, keys):
        comm = self.comm

        keys = np.asarray(keys, dtype=int)

        # Route query keys to their home ranks
        home = home_rank(keys, comm.size)
        sord = np.argsort(home)
        scount = np.bincount(home, minlength=comm.size)

        recv, (rcount, _) = self._alltoallcv(comm, keys[sord], scount)

        # Look up owner ranks in the sorted table
        ans = self.ranks[np.searchsorted(self.keys, recv)]

        # Send answers back; rcount mirrors the forward counts
        ret, _ = self._alltoallcv(comm, ans, rcount)

        # Unshuffle from home-rank order back to caller order
        result = np.empty_like(keys)
        result[sord] = ret
        return result


class AlltoallFuture:
    def __init__(self, parent, nsend, nrecv, shape, dtype, scountdisps,
                 rcountdisps, rinv):
        self._parent = parent
        self._rinv = rinv

        # Preallocate send and receive buffers
        self._svals = np.empty((nsend, *shape), dtype=dtype)
        self._rvals = np.empty((nrecv, *shape), dtype=dtype)

        # Create persistent request
        sbuf = (self._svals, scountdisps)
        rbuf = (self._rvals, rcountdisps)
        self._req = autofree(parent._alltoallv_init(parent.comm, sbuf, rbuf))

    def start(self, dset):
        self._parent._prepare_sendbuf(dset, self._svals)
        self._req.Start()
        return self

    def test(self):
        return self._req.Test()

    def wait(self):
        self._req.Wait()
        return self._rvals[self._rinv]


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

    def _prepare_sendbuf(self, dset, out):
        np.take(dset[self.start:self.end], self.sidx, axis=0, out=out,
                mode='clip')

    def future(self, shape, dtype):
        return AlltoallFuture(self, len(self.sidx), self.cnt, shape, dtype,
                              self.bcountdisps, self.acountdisps, self.rinv)

    def __call__(self, dset):
        return self.future(dset.shape[1:], dset.dtype).start(dset).wait()


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

    def _prepare_sendbuf(self, dset, out):
        np.take(dset, self.sinv, axis=0, out=out, mode='clip')

    def future(self, shape, dtype):
        return AlltoallFuture(self, len(self.sinv), self.cnt, shape, dtype,
                              self.acountdisps, self.bcountdisps, self.rinv)

    def __call__(self, dset):
        return self.future(dset.shape[1:], dset.dtype).start(dset).wait()


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
        sp = np.searchsorted(bidx, minmax)
        dcount = sp[1::2] - sp[::2]

        # Exchange indices
        eidx, (_, edisps) = self._alltoallcv(comm, bidx, dcount, sp[::2])

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

    def _prepare_sendbuf(self, dset, out):
        np.take(dset[self.start:self.end], self.sidx, axis=0, out=out,
                mode='clip')

    def future(self, shape, dtype):
        return AlltoallFuture(self, len(self.sidx), self.cnt, shape, dtype,
                              self.scountdisps, self.rcountdisps, ...)

    def __call__(self, dset):
        return self.future(dset.shape[1:], dset.dtype).start(dset).wait()


class Sorter(AlltoallMixin):
    typemap = {
        'int8': np.uint8, 'int16': np.uint16,
        'int32': np.uint32, 'int64': np.uint64,
        'float32': np.int32, 'float64': np.int64
    }

    def __init__(self, comm, keys):
        self.comm = comm

        # Locally sort our outbound keys
        self.sidx = np.argsort(keys)
        skeys = keys[self.sidx]

        # Determine the total size of the array
        size = scal_coll(comm.Allreduce, len(keys))

        self.start, end, csize = get_start_end_csize(comm, size)
        self.cnt = end - self.start

        # Determine what to send to each rank
        sdisps = self._splitters(skeys, self.start)
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
            udtype = self.typemap[dtype.name]
            return skeys.view(udtype) ^ udtype(np.iinfo(dtype).max + 1)
        elif np.issubdtype(dtype, np.floating):
            shift = 8*dtype.itemsize - 1
            idtype = self.typemap[dtype.name]
            udtype = self.typemap[np.dtype(idtype).name]

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

        e, rt = kmin, 0
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

        # Count the occurrences of each probe in skeys
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

    @property
    def argidx(self):
        svals = self.start + np.argsort(self.ridx)
        rvals = np.empty(len(self.sidx), dtype=svals.dtype)

        self._alltoallv(self.comm, (svals, self.rcountdisps),
                        (rvals, self.scountdisps))

        return rvals[np.argsort(self.sidx)]


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
