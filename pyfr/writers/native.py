import contextlib
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
from pyfr.mpiutil import Gatherer, get_comm_rank_root, mpi, scal_coll
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
    MAGIC_LUSTRE = 0x0bd00bd0
    O_LOV_DELAY_CREATE = 0x1002100

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

        # If we are on a Lustre filesystem
        self.on_lustre_fs = self._on_lustre_fs(basedir)

        # Current asynchronous writing operation (if any)
        self._awriter = None

    @staticmethod
    def from_integrator(intg, basedir, basename, prefix, *, extn='.pyfrs'):
        return NativeWriter(intg.system.mesh, intg.cfg, intg.backend.fpdtype,
                            basedir, basename, prefix=prefix,
                            isrestart=intg.isrestart)

    @classmethod
    def _on_lustre_fs(cls, basedir):
        if sys.platform == 'linux' and 'PYFR_DISABLE_LUSTRE' not in os.environ:
            buf = (ctypes.c_int * 128)()
            get_libc_function('statfs')(basedir.encode(), buf)
            return buf[0] == cls.MAGIC_LUSTRE
        else:
            return False

    @contextlib.contextmanager
    def _create_file(self, path):
        if self.on_lustre_fs:
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
            except (IOError, OSError):
                pass

        with h5py.File(path, 'w', libver='latest') as f:
            yield f

    def _open_file(self, path):
        f = open(path, 'r+b')

        # If we are on a Lustre file system then take a group lock
        if self.on_lustre_fs:
            fcntl.ioctl(f.fileno(), self.LL_IOC_GROUP_LOCK, self.LL_GROUP)

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

        # Delegate to _write to do the actual outputting
        f, th = self._write(self.tname, gdata, metadata, async_=async_)

        # Determine the final output path
        path = self.fgen.send(tcurr)

        def oncomplete():
            # Close the file
            f.close()

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

    def _prepare_file(self, path, metadata):
        doffs = {}

        with self._create_file(path) as f:
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

        return doffs

    def _write(self, path, data, metadata, *, async_=True):
        comm, rank, root = get_comm_rank_root()

        # Have the root rank prepare the output file
        if rank == root:
            doffs = self._prepare_file(path, metadata)
        else:
            doffs = None

        # Distrbute the offsets of each dataset
        doffs = comm.bcast(doffs, root=root)

        # Open the file for writing
        f = self._open_file(path)

        # Helper function for writing arrays at offsets
        def write_off(k, v, n):
            if len(v):
                f.seek(doffs[k] + n*(v.nbytes // len(v)))
                v.tofile(f)

        # Main writing function
        def write(einfo):
            for ek, (gatherer, subset, *_) in einfo.items():
                write_off(ek, data[ek], gatherer.off)
                write_off(f'{ek}-parts', gatherer.rsrc, gatherer.off)

                # If the element has been subset then write the index data
                if subset:
                    write_off(f'{ek}-idxs', gatherer.ridx, gatherer.off)

        if async_:
            th = threading.Thread(target=write, args=(self._einfo,))
            th.start()
            return f, th
        else:
            write(self._einfo)
            return f, None


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
