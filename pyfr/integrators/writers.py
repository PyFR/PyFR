# -*- coding: utf-8 -*-

from abc import abstractmethod
import itertools as it
import os

import numpy as np

from pyfr.integrators.base import BaseIntegrator
from pyfr.mpiutil import get_comm_rank_root
from pyfr.util import rm


class BaseWriter(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super(BaseWriter, self).__init__(*args, **kwargs)

        # Base output directory and file name
        self._basedir = self._cfg.getpath('soln-output', 'basedir', '.')
        self._basename = self._cfg.get('soln-output', 'basename', raw=True)

        # Output counter (incremented each time output() is called)
        self.nout = 0

    def output(self, solnmap, stats):
        comm, rank, root = get_comm_rank_root()

        # Convert the config and stats objects to strings
        if rank == root:
            metadata = dict(config=self._cfg.tostr(),
                            stats=stats.tostr(),
                            mesh_uuid=self._mesh_uuid)
        else:
            metadata = None

        # Determine the output path
        path = self._get_output_path()

        # Delegate to _write to do the actual outputting
        self._write(path, solnmap, metadata)

        # Increment the output number
        self.nout += 1

    @abstractmethod
    def _write(self, path, solnmap, metadata):
        pass

    def _get_output_path(self):
        # Substitute %(t) and %(n) for the current time and output number
        fname = self._basename % dict(t=self.tcurr, n=self.nout)

        # Append the '.pyfrs' extension
        if not fname.endswith('.pyfrs'):
            fname += '.pyfrs'

        return os.path.join(self._basedir, fname)

    def _get_name_for_soln(self, etype, prank=None):
        prank = prank or self._rallocs.prank
        return 'soln_{}_p{}'.format(etype, prank)


class FileWriter(BaseWriter):
    writer_name = 'pyfrs-file'

    def __init__(self, *args, **kwargs):
        super(FileWriter, self).__init__(*args, **kwargs)

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the type and shape of each element in the partition
        etypes, shapes = self._system.ele_types, self._system.ele_shapes

        # Gather this information onto the root rank
        eleinfo = comm.gather(zip(etypes, shapes), root=root)

        if rank == root:
            self._mpi_rbufs = mpi_rbufs = []
            self._mpi_rreqs = mpi_rreqs = []
            self._mpi_names = mpi_names = []
            self._loc_names = loc_names = []

            for mrank, meleinfo in enumerate(eleinfo):
                prank = self._rallocs.mprankmap[mrank]
                for tag, (etype, dims) in enumerate(meleinfo):
                    name = self._get_name_for_soln(etype, prank)

                    if mrank == root:
                        loc_names.append(name)
                    else:
                        rbuf = np.empty(dims, dtype=self._backend.fpdtype)
                        rreq = comm.Recv_init(rbuf, mrank, tag)

                        mpi_rbufs.append(rbuf)
                        mpi_rreqs.append(rreq)
                        mpi_names.append(name)

    def _write(self, path, solnmap, metadata):
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
            names = it.chain(self._loc_names, self._mpi_names)
            solns = it.chain(solnmap.values(), self._mpi_rbufs)

            # Create the output dictionary
            outdict = dict(zip(names, solns), **metadata)

            with open(path, 'wb') as f:
                np.savez(f, **outdict)


class DirWriter(BaseWriter):
    writer_name = 'pyfrs-dir'

    def _write(self, path, solnmap, metadata):
        comm, rank, root = get_comm_rank_root()

        # Create the output directory and save the config/status files
        if rank == root:
            if os.path.exists(path):
                rm(path)

            os.mkdir(path)

            # Write out our metadata
            for name, data in metadata.items():
                np.save(os.path.join(path, name), data)

        # Wait for this to complete
        comm.barrier()

        # Save the solutions
        for etype, buf in solnmap.items():
            solnpath = os.path.join(path, self._get_name_for_soln(etype))
            np.save(solnpath, buf)
