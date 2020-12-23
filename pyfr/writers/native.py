# -*- coding: utf-8 -*-

import os
import re

import h5py
import numpy as np

from pyfr.mpiutil import get_comm_rank_root


def write_pyfrms(path, data):
    # Save to disk
    with h5py.File(path, 'w') as f:
        for k in filter(lambda k: isinstance(k, str), data):
            f[k] = data[k]

        for p, q in filter(lambda k: isinstance(k, tuple), data):
            f[p].attrs[q] = data[p, q]


class NativeWriter(object):
    def __init__(self, intg, mdata, basedir, basename, *, extn='.pyfrs'):
        # Base output directory and file name
        self.basedir = basedir
        self.basename = basename

        # Append the relevant extension
        if not self.basename.endswith(extn):
            self.basename += extn

        # Output counter (incremented each time write() is called)
        self.nout = self._restore_nout() if intg.isrestart else 0

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Gather the output metadata across all ranks
        mdata = comm.allgather(mdata)

        # Parallel I/O
        if (h5py.get_config().mpi and
            'PYFR_FORCE_SERIAL_HDF5' not in os.environ):
            self._write = self._write_parallel
            self._loc_names = loc_names = []
            self._global_shape_list = []

            for mrank, mfields in enumerate(mdata):
                prank = intg.rallocs.mprankmap[mrank]

                # Loop over all element types across all ranks
                for fname, fshape, fdtype in mfields:
                    name = f'{fname}_p{prank}'
                    self._global_shape_list.append((name, fshape, fdtype))

                    if rank == mrank:
                        loc_names.append(name)
        # Serial I/O
        else:
            self._write = self._write_serial

            if rank == root:
                self._loc_info = loc_info = []
                self._mpi_info = mpi_info = []

                for mrank, mfields in enumerate(mdata):
                    prank = intg.rallocs.mprankmap[mrank]
                    for fname, fshape, fdtype in mfields:
                        name = f'{fname}_p{prank}'

                        if mrank == root:
                            loc_info.append(name)
                        else:
                            mpi_info.append((name, mrank, fshape, fdtype))

    def write(self, data, metadata, tcurr):
        # Determine the output path
        path = self._get_output_path(tcurr)

        # Delegate to _write to do the actual outputting
        self._write(path, data, metadata)

        # Increment the output number
        self.nout += 1

        # Return the path
        return path

    def _restore_nout(self):
        nout = 0

        # See if the basename appears to depend on {n}
        if re.search('{n[^}]*}', self.basename):
            # Quote and substitute
            bn = re.escape(self.basename)
            bn = re.sub(r'\\{n[^}]*\\}', r'(\\s*\\d+\\s*)', bn)
            bn = re.sub(r'\\{t[^}]*\\}', r'(?:.*?)', bn) + '$'

            for f in os.listdir(self.basedir):
                m = re.match(bn, f)
                if m:
                    nout = max(nout, int(m.group(1)) + 1)

        return nout

    def _get_output_path(self, tcurr):
        # Substitute {t} and {n} for the current time and output number
        fname = self.basename.format(t=tcurr, n=self.nout)

        return os.path.join(self.basedir, fname)

    def _write_parallel(self, path, data, metadata):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(path, 'w', driver='mpio', comm=comm) as f:
            dmap = {}
            for name, shape, dtype in self._global_shape_list:
                dmap[name] = f.create_dataset(
                    name, shape, dtype=dtype
                )

            # Write out our data sets using 2 GiB chunks
            for name, dat in zip(self._loc_names, data):
                nrows = len(dat)
                rowsz = dat.nbytes // nrows
                rstep = 2*1024**3 // rowsz

                if rstep == 0:
                    raise RuntimeError('Array is too large for parallel I/O')

                for ix in range(0, nrows, rstep):
                    dmap[name][ix:ix + rstep] = dat[ix:ix + rstep]

            # Metadata information has to be transferred to all the ranks
            if rank == root:
                mmap = [(k, len(v.encode()))
                        for k, v in metadata.items()]
            else:
                mmap = None

            for name, size in comm.bcast(mmap, root=root):
                d = f.create_dataset(name, (), dtype=f'S{size}')

                if rank == root:
                    d.write_direct(np.array(metadata[name], dtype='S'))

        # Wait for everyone to finish writing
        comm.barrier()

    def _write_serial(self, path, data, metadata):
        comm, rank, root = get_comm_rank_root()

        if rank != root:
            for tag, buf in enumerate(data):
                comm.Send(buf.copy(), root)
        else:
            with h5py.File(path, 'w') as f:
                # Write the metadata
                for k, v in metadata.items():
                    f[k] = np.array(v, dtype='S')

                # Write our local data
                for k, v in zip(self._loc_info, data):
                    f[k] = v

                # Receive and write the remote data
                for k, mrank, shape, dtype in self._mpi_info:
                    v = np.empty(shape, dtype=dtype)
                    comm.Recv(v, mrank)

                    f[k] = v

        # Wait for the root rank to finish writing
        comm.barrier()
