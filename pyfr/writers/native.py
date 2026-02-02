import ctypes
import errno
import fcntl
import os
import struct
import sys
import threading
import time
import uuid

import h5py
import numpy as np

from pyfr._version import __version__
from pyfr.ctypesutil import get_libc_function
from pyfr.mpiutil import Gatherer, autofree, get_comm_rank_root, mpi, scal_coll
from pyfr.quadrules import get_quadrule
from pyfr.shapes import BaseShape
from pyfr.util import file_path_gen, mv, subclass_where


class NativeWriter:
    # Lustre constants
    LL_IOC_GROUP_LOCK = 0x4008669e
    LL_GROUP = 0x1412
    LL_IOC_LOV_SETSTRIPE = 0x4008669a
    LOV_USER_MAGIC_V1 = 0x0bd10bd0
    LOV_USER_MAGIC_V3 = 0x0bd30bd0
    O_LOV_DELAY_CREATE = 0x1002100

    # Lock to serialise writes across all instances
    _lock = threading.Lock()

    def __init__(self, mesh, cfg, fpdtype, basedir, basename, prefix, *,
                 extn='.pyfrs', isrestart=False):
        comm, rank, root = get_comm_rank_root()

        self.cfg = cfg
        self.prefix = prefix
        self.fpdtype = fpdtype

        # Tally up how many elements of each type our partition has
        self._ecounts = {etype: len(mesh.eidxs.get(etype, []))
                         for etype in mesh.etypes}

        # Append the relevant extension
        if not basename.endswith(extn):
            basename += extn

        # Output counter (incremented each time write() is called)
        self.fgen = file_path_gen(basedir, basename, isrestart)

        # Temporary file name
        if rank == root:
            tname = os.path.join(basedir, f'pyfr-{uuid.uuid4()}{extn}')
        else:
            tname = None

        self.tname = comm.bcast(tname, root=root)

        # Query the file system type
        self.fstype = self._get_fstype(basedir)

        # Determine if to perform serial or parallel writes
        if self.fstype == 'lustre':
            self._get_writefn = self._get_writefn_parallel
        else:
            self._get_writefn = self._get_writefn_serial

            # Private communicator for serial writes
            self._scomm = autofree(comm.Dup())

        # Current asynchronous writing operation (if any)
        self._awriter = None

    @staticmethod
    def from_integrator(intg, basedir, basename, prefix, *, extn='.pyfrs',
                        fpdtype=None):
        _ftype = fpdtype or intg.backend.fpdtype
        return NativeWriter(intg.system.mesh, intg.cfg, _ftype, basedir,
                            basename, prefix=prefix, isrestart=intg.isrestart)

    @staticmethod
    def _get_fstype(basedir):
        if sys.platform == 'linux':
            fstypes = {
                0x00006969: 'nfs',
                0x0bd00bd0: 'lustre'
            }

            buf = (ctypes.c_int * 128)()
            get_libc_function('statfs')(str(basedir).encode(), buf)

            return fstypes.get(buf[0], None)

        return None

    def _create_file(self, path):
        comm, rank, root = get_comm_rank_root()

        if self.fstype == 'lustre' and rank == root:
            # Lustre pool name
            pool = None

            # Stripe size, count, and offset
            ssize = 128*1024**2
            scount = 2**16 - 1
            soff = 2**16 - 1

            magic = self.LOV_USER_MAGIC_V3 if pool else self.LOV_USER_MAGIC_V1
            flags = (os.O_CREAT | os.O_EXCL | os.O_WRONLY
                     | self.O_LOV_DELAY_CREATE)

            try:
                # Create the file with the special delay-create flag
                try:
                    fd = os.open(path, flags)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        os.remove(path)
                        fd = os.open(path, flags)
                    else:
                        raise

                # Issue an ioctl to set the stripe parameters
                try:
                    arg = struct.pack('=IIQQIHH', magic, 1, 0, 0, ssize,
                                      scount, soff)
                    if pool:
                        arg += pool.ljust(16, '\0').encode()

                    fcntl.ioctl(fd, self.LL_IOC_LOV_SETSTRIPE, arg)
                finally:
                    os.close(fd)
            except OSError:
                pass

    def _open_file(self, path):
        f = open(path, 'r+b')

        return f

    def set_shapes_eidxs(self, shapes, eidxs):
        comm, rank, root = get_comm_rank_root()

        # Prepare the element information
        self._einfo = {}
        for etype, ecount in self._ecounts.items():
            # See if any ranks want to write elements of this type
            eshape = comm.allgather(shapes.get(etype))
            if any(eshape):
                # Create a gatherer for this element type
                idxs = eidxs.get(etype, [])
                gatherer = Gatherer(comm, idxs)

                # Exchange counts and offsets
                noff = comm.allgather((gatherer.cnt, gatherer.off))
                noff = {i: j for i, j in enumerate(noff) if j}

                # Determine the final shape of the element array
                shape = (gatherer.tot, *next(es for es in eshape if es))

                # Determine the polynomial order
                ecls = subclass_where(BaseShape, name=etype)
                order = ecls.order_from_npts(shape[2])

                # See if the element is being subset
                subset = comm.allreduce(len(idxs) != ecount, op=mpi.LOR)

                # Also get the associated nodal points
                rname = self.cfg.get(f'solver-elements-{etype}', 'soln-pts')
                upts = get_quadrule(etype, rname, shape[2]).pts

                ek = f'p{order}-{etype}'
                self._einfo[ek] = (gatherer, subset, shape, etype, noff, upts)

    def probe(self):
        if self._awriter is not None and self._awriter.test():
            self._awriter = None

    def flush(self):
        if self._awriter is not None:
            self._awriter.wait()
            self._awriter = None

    def write(self, data, tcurr, metadata=None, timeout=0, callback=None):
        async_ = bool(timeout)
        comm, rank, root = get_comm_rank_root()

        # Wait for any existing write operations to finish
        if self._awriter is not None:
            self._awriter.wait()
            self._awriter = None

        if metadata:
            if rank != root:
                raise ValueError('Metadata must be written by the root rank')

            metadata = dict(metadata, creator=f'pyfr {__version__}', version=1)

            # Convert all strings to arrays
            for k, v in metadata.items():
                if isinstance(v, str):
                    metadata[k] = np.array(v, dtype='S')

        # Gather the solution data into contiguous arrays
        gdata = {}
        for ek, (gatherer, subset, shape, etype, *_) in self._einfo.items():
            if etype in data:
                dset = data[etype]
            else:
                dset = np.empty((0, *shape[1:]), dtype=self.fpdtype)

            gdata[ek] = gatherer(dset)

        # Prepare the output file
        doffs = self._prepare_file(self.tname, metadata)

        # Obtain the relevant low-level writer function
        wfn = self._get_writefn(self.tname, gdata, doffs)

        # Perform the file writing
        if async_:
            th = threading.Thread(target=wfn)
            th.start()
        else:
            wfn()

        # Determine the final output path
        path = self.fgen.send(tcurr)

        def oncomplete():
            # Have the root rank move it into place
            if rank == root:
                mv(self.tname, path)

            # Fire off any user-provided callback
            if callback is not None:
                callback(path)

        if async_:
            self._awriter = _AsyncCompleter(th, timeout, oncomplete)
        else:
            oncomplete()

    def _iter_bufs(self, data, callback):
        for ek, (gatherer, subset, *_) in self._einfo.items():
            callback(ek, data[ek], gatherer.off)
            callback(f'{ek}-parts', gatherer.rsrc, gatherer.off)

            # If the element has been subset then write the index data
            if subset:
                callback(f'{ek}-idxs', gatherer.ridx, gatherer.off)

    def _get_writefn_parallel(self, path, data, doffs):
        # Open the file for writing
        f = self._open_file(path)

        # Callback to write out individual arrays at offsets
        def write_off(k, v, n):
            if len(v):
                f.seek(doffs[k] + n*(v.nbytes // len(v)))
                f.write(memoryview(np.ascontiguousarray(v)))

        # Main writing function
        def write():
            with self._lock:
                self._iter_bufs(data, write_off)

                # Close the file
                f.close()

        return write

    def _get_writefn_serial(self, path, data, doffs):
        comm, rank, root = get_comm_rank_root()
        wbufs = []

        # Helper function for extracting data about buffers
        def add_buf(k, v, n):
            if len(v):
                wbufs.append((doffs[k] + n*(v.nbytes // len(v)), v.nbytes, v))

        # Determine what buffers we need to write
        self._iter_bufs(data, add_buf)

        # Sort our buffers by their offset
        wbufs.sort()

        # Collate this data to the root rank
        bufs = comm.gather([(off, nb) for off, nb, _ in wbufs], root=root)

        if rank == root:
            # Open the file for writing
            f = self._open_file(path)

            # Prepare the final buffer list
            iwbufs = (b for _, _, b in wbufs)
            bufs = [(*rbinfo, r)
                    for r, rbufs in enumerate(bufs)
                    for rbinfo in rbufs]

            def write():
                with self._lock:
                    # Write the buffers to the file in order
                    for off, nb, r in sorted(bufs):
                        if r == root:
                            b = next(iwbufs)
                        else:
                            b = np.empty(nb, dtype=np.uint8)
                            self._scomm.Recv(b, r)

                        f.seek(off)
                        f.write(memoryview(np.ascontiguousarray(b)))

        else:
            def write():
                with self._lock:
                    # Send each of our buffers to the root rank for writing
                    for _, _, b in wbufs:
                        self._scomm.Send(
                            np.ascontiguousarray(b.view(np.uint8)), root
                        )

        return write

    def _prepare_file(self, path, metadata):
        comm, rank, root = get_comm_rank_root()

        # Perform any pre-creation activities
        self._create_file(path)

        # Have the root rank lay down the structure of the file
        if rank == root:
            doffs = {}

            with h5py.File(path, 'w', libver='latest') as f:
                # Write the metadata
                for k, v in metadata.items():
                    f[k] = v

                # Create the datasets
                g = f.create_group(self.prefix)
                for ek, (gatherer, subset, shape, *_) in self._einfo.items():
                    g.create_dataset(ek, shape, self.fpdtype)
                    g.create_dataset(f'{ek}-parts', shape[0:1], np.int32)

                    if subset:
                        g.create_dataset(f'{ek}-idxs', shape[0:1], np.int64)

                # Add each elements nodal points as an attribute
                for ek, (*_, upts) in self._einfo.items():
                    g[ek].attrs['pts'] = upts

                # Obtain the offsets of these datasets
                for k, v in g.items():
                    v[(-1,)*v.ndim] = 0
                    doffs[k] = v.id.get_offset()

            return comm.bcast(doffs, root=root)
        else:
            return comm.bcast(None, root=root)


class _AsyncCompleter:
    def __init__(self, th, timeout, callback):
        self.th = th
        self.timeout = timeout
        self.callback = callback

        self.done = False
        self.start = time.time()

    def _test_with_timeout(self, timeout):
        comm, rank, root = get_comm_rank_root()

        if not self.done:
            if time.time() - self.start >= timeout:
                self.th.join()

            self.done = not self.th.is_alive()

        # See if everyone is done
        if scal_coll(comm.Allreduce, int(self.done), op=mpi.LAND):
            self.callback()

            return True
        else:
            return False

    def test(self):
        return self._test_with_timeout(self.timeout)

    def wait(self):
        return self._test_with_timeout(0.0)
