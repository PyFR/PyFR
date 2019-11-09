# -*- coding: utf-8 -*-

import itertools as it
import os
import re

import h5py
import numpy as np

from pyfr.mpiutil import get_comm_rank_root


class NativeWriter(object):
    def __init__(self, intg, nvars, basedir, basename, *, prefix,
                 extn='.pyfrs'):
        # Base output directory and file name
        self.basedir = basedir
        self.basename = basename

        # Append the relevant extension
        if not self.basename.endswith(extn):
            self.basename += extn

        # Prefix given to each data array in the output file
        self.prefix = prefix

        # Output counter (incremented each time write() is called)
        self.nout = self._restore_nout() if intg.isrestart else 0

        # Copy the float type
        self.fpdtype = intg.backend.fpdtype

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the type and shape of each element in the partition
        etypes = intg.system.ele_types
        shapes = [(nupts, nvars, neles)
                  for nupts, _, neles in intg.system.ele_shapes]

        # Gather
        eleinfo = comm.allgather(zip(etypes, shapes))

        # Parallel I/O
        if (h5py.get_config().mpi and
            'PYFR_FORCE_SERIAL_HDF5' not in os.environ):
            self._write = self._write_parallel
            self._loc_names = loc_names = []
            self._global_shape_list = []

            for mrank, meleinfo in enumerate(eleinfo):
                prank = intg.rallocs.mprankmap[mrank]

                # Loop over all element types across all ranks
                for etype, shape in meleinfo:
                    name = self._get_name_for_data(etype, prank)
                    self._global_shape_list.append((name, shape))

                    if rank == mrank:
                        loc_names.append(name)
        # Serial I/O
        else:
            self._write = self._write_serial

            if rank == root:
                self._loc_info = loc_info = []
                self._mpi_info = mpi_info = []

                for mrank, meleinfo in enumerate(eleinfo):
                    prank = intg.rallocs.mprankmap[mrank]
                    for tag, (etype, shape) in enumerate(meleinfo):
                        name = self._get_name_for_data(etype, prank)

                        if mrank == root:
                            loc_info.append(name)
                        else:
                            mpi_info.append((name, mrank, tag, shape))

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
            bn = re.sub(r'\\{n[^}]*\\}', r'(\s*\d+\s*)', bn)
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

    def _get_name_for_data(self, etype, prank):
        return '{}_{}_p{}'.format(self.prefix, etype, prank)

    def _write_parallel(self, path, data, metadata):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(path, 'w', driver='mpio', comm=comm) as f:
            dmap = {}
            for name, shape in self._global_shape_list:
                dmap[name] = f.create_dataset(
                    name, shape, dtype=self.fpdtype
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
                d = f.create_dataset(name, (), dtype='S{}'.format(size))

                if rank == root:
                    d.write_direct(np.array(metadata[name], dtype='S'))

        # Wait for everyone to finish writing
        comm.barrier()

    def _write_serial(self, path, data, metadata):
        comm, rank, root = get_comm_rank_root()

        if rank != root:
            for tag, buf in enumerate(data):
                comm.Send(buf.copy(), root, tag)
        else:
            with h5py.File(path, 'w') as f:
                # Write the metadata
                for k, v in metadata.items():
                    f[k] = np.array(v, dtype='S')

                # Write our local data
                for k, v in zip(self._loc_info, data):
                    f[k] = v

                # Receive and write the remote data
                for k, mrank, tag, shape in self._mpi_info:
                    v = np.empty(shape, dtype=self.fpdtype)
                    comm.Recv(v, mrank, tag)

                    f[k] = v

        # Wait for the root rank to finish writing
        comm.barrier()
