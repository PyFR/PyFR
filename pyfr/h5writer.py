# -*- coding: utf-8 -*-

import itertools as it
import os

import h5py
import numpy as np

from pyfr.mpiutil import get_comm_rank_root


class H5Writer(object):
    def __init__(self, intg, basedir, basename, prefix):
        # Base output directory and file name and prefix for the data-set.
        self._basedir = basedir
        self._basename = basename
        self._prefix = prefix

        # Output counter (incremented each time write() is called)
        self.nout = 0

        # Copy the float type
        self.fpdtype = intg.backend.fpdtype

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the type and shape of each element in the partition
        etypes, shapes = intg.system.ele_types, intg.system.ele_shapes

        # Gather this information and distribute
        eleinfo = comm.allgather(zip(etypes, shapes))

        # Deciding if parallel
        parallel = (h5py.get_config().mpi and
                    h5py.version.version_tuple >= (2, 5) and
                    'PYFR_FORCE_SERIAL_HDF5' not in os.environ)

        if parallel:
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
        else:
            self._write = self._write_serial

            if rank == root:
                self._mpi_rbufs = mpi_rbufs = []
                self._mpi_rreqs = mpi_rreqs = []
                self._mpi_names = mpi_names = []
                self._loc_names = loc_names = []

                for mrank, meleinfo in enumerate(eleinfo):
                    prank = intg.rallocs.mprankmap[mrank]
                    for tag, (etype, shape) in enumerate(meleinfo):
                        name = self._get_name_for_data(etype, prank)

                        if mrank == root:
                            loc_names.append(name)
                        else:
                            rbuf = np.empty(shape, dtype=self.fpdtype)
                            rreq = comm.Recv_init(rbuf, mrank, tag)

                            mpi_rbufs.append(rbuf)
                            mpi_rreqs.append(rreq)
                            mpi_names.append(name)

    def write(self, data, metadata, tcurr):
        # Determine the output path
        path = self._get_output_path(tcurr)

        # Delegate to _write to do the actual outputting
        self._write(path, data, metadata)

        # Increment the output number
        self.nout += 1

    def _get_output_path(self, tcurr):
        # Substitute %(t) and %(n) for the current time and output number
        fname = self._basename % dict(t=tcurr, n=self.nout)

        # Append the '.pyfrs' extension
        if not fname.endswith('.pyfrs'):
            fname += '.pyfrs'

        return os.path.join(self._basedir, fname)

    def _get_name_for_data(self, etype, prank):
        return '{}_{}_p{}'.format(self._prefix, etype, prank)

    def _write_parallel(self, path, data, metadata):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(path, 'w', driver='mpio', comm=comm) as h5file:
            dmap = {}
            for name, shape in self._global_shape_list:
                dmap[name] = h5file.create_dataset(
                    name, shape, dtype=self.fpdtype
                )

            for s, dat in zip(self._loc_names, data):
                dmap[s][:] = dat

            # Metadata information has to be transferred to all the ranks
            if rank == root:
                mmap = [(k, len(v.encode()))
                        for k, v in metadata.items()]
            else:
                mmap = None

            for name, size in comm.bcast(mmap, root=root):
                d = h5file.create_dataset(name, (), dtype='S{}'.format(size))

                if rank == root:
                    d.write_direct(np.array(metadata[name], dtype='S'))

    def _write_serial(self, path, data, metadata):
        from mpi4py import MPI

        comm, rank, root = get_comm_rank_root()

        if rank != root:
            for tag, buf in enumerate(data):
                comm.Send(buf.copy(), root, tag)
        else:
            # Recv all of the non-local data
            MPI.Prequest.Startall(self._mpi_rreqs)
            MPI.Prequest.Waitall(self._mpi_rreqs)

            # Combine local and MPI data
            names = it.chain(self._loc_names, self._mpi_names)
            dats = it.chain(data, self._mpi_rbufs)

            # Convert any metadata to ASCII
            metadata = {k: np.array(v, dtype='S')
                        for k, v in metadata.items()}

            # Create the output dictionary
            outdict = dict(zip(names, dats), **metadata)

            with h5py.File(path, 'w') as h5file:
                for k, v in outdict.items():
                    h5file[k] = v
