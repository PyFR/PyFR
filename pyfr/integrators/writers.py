# -*- coding: utf-8 -*-

from collections import OrderedDict
import itertools as it
import os

import h5py
import numpy as np

from pyfr.integrators.base import BaseIntegrator
from pyfr.mpiutil import get_comm_rank_root


class H5Writer(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Base output directory and file name
        self._basedir = self.cfg.getpath('soln-output', 'basedir', '.')
        self._basename = self.cfg.get('soln-output', 'basename', raw=True)

        # Output counter (incremented each time output() is called)
        self.nout = 0

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the type and shape of each element in the partition
        etypes, shapes = self.system.ele_types, self.system.ele_shapes

        # Gather this information onto the root rank
        eleinfo = comm.gather(tuple(zip(etypes, shapes)), root=root)

        # Deciding if parallel
        parallel = (h5py.get_config().mpi and
                    h5py.version.version_tuple >= (2, 5) and
                    not self.cfg.getbool('soln-output', 'serial-h5', False))

        if rank == root:
            sollist = OrderedDict()
            for mrank, meleinfo in enumerate(eleinfo):
                prank = self.rallocs.mprankmap[mrank]
                for etype, dims in meleinfo:
                    sollist[etype, prank] = \
                            (self._get_name_for_soln(etype, prank), dims)

        else:
            sollist = None

        self.sollist = comm.bcast(sollist, root=root)

        if parallel:
            self._write = self._write_parallel

        else:
            self._write = self._write_serial

            if rank == root:
                self._mpi_rbufs = mpi_rbufs = []
                self._mpi_rreqs = mpi_rreqs = []

                for mrank, meleinfo in enumerate(eleinfo):
                    prank = self.rallocs.mprankmap[mrank]
                    for tag, (etype, dims) in enumerate(meleinfo):
                        if mrank != root:
                            rbuf = np.empty(dims, dtype=self.backend.fpdtype)
                            rreq = comm.Recv_init(rbuf, mrank, tag)

                            mpi_rbufs.append(rbuf)
                            mpi_rreqs.append(rreq)

    def output(self, solnmap, stats):
        comm, rank, root = get_comm_rank_root()

        # Convert the config and stats objects to strings
        if rank == root:
            metadata = dict(config=self.cfg.tostr(),
                            stats=stats.tostr(),
                            mesh_uuid=self._mesh_uuid)
        else:
            metadata = None

        # Determine the output path
        path = self._get_output_path()

        # Delegate to _write to do the actual outputting
        self._write(path, solnmap, self.sollist, metadata)

        # Increment the output number
        self.nout += 1

    def _get_output_path(self):
        # Substitute %(t) and %(n) for the current time and output number
        fname = self._basename % dict(t=self.tcurr, n=self.nout)

        # Append the '.pyfrs' extension
        if not fname.endswith('.pyfrs'):
            fname += '.pyfrs'

        return os.path.join(self._basedir, fname)

    def _get_name_for_soln(self, etype, prank=None):
        prank = self.rallocs.prank if prank is None else prank
        return 'soln_{}_p{}'.format(etype, prank)

    def _write_parallel(self, path, solnmap, sollist, metadata):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(path, 'w', driver='mpio', comm=comm) as h5file:
            smap = {}
            for name, shape in sollist.values():
                smap[name] = h5file.create_dataset(
                    name, shape, dtype=self.backend.fpdtype
                )

            for e, sol in solnmap.items():
                s = sollist[e, self.rallocs.prank][0]
                smap[s][:] = sol

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

    def _write_serial(self, path, solnmap, sollist, metadata):
        from mpi4py import MPI

        comm, rank, root = get_comm_rank_root()

        if rank != root:
            for tag, buf in enumerate(solnmap.values()):
                comm.Send(buf.copy(), root, tag)
        else:
            # Recv all of the non-local solution mats
            MPI.Prequest.Startall(self._mpi_rreqs)
            MPI.Prequest.Waitall(self._mpi_rreqs)

            # Combine local and MPI data
            names = [name for name, _ in sollist.values()]
            solns = it.chain(solnmap.values(), self._mpi_rbufs)

            # Convert any metadata to ASCII
            metadata = {k: np.array(v, dtype='S')
                        for k, v in metadata.items()}

            # Create the output dictionary
            outdict = dict(zip(names, solns), **metadata)

            with h5py.File(path, 'w') as h5file:
                for k, v in outdict.items():
                    h5file[k] = v
