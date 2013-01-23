# -*- coding: utf-8 -*-

import os
import itertools
import shutil

from abc import abstractmethod

from mpi4py import MPI

import numpy as np

from pyfr.integrators.base import BaseIntegrator
from pyfr.util import get_comm_rank_root

class BaseWriter(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super(BaseWriter, self).__init__(*args, **kwargs)

        # Output base directory
        self._basedir = self._cfg.getpath('soln-output', 'basedir', '.')

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
        # Get the output directory
        d = self._basedir

        # Current time and output number
        t = format(self.tcurr)
        n = format(self.nout)

        # File/dir to write the solution to
        f = self._cfg.get('soln-output', 'basename', vars=dict(t=t, n=n))

        return os.path.join(d, f + '.pyfrs')

    def _get_name_for_soln(self, type, prank=None):
        prank = prank or self._rallocs.prank
        return 'soln_%s_p%d' % (type, prank)


class FileWriter(BaseWriter):
    writer_name = 'pyfrs-file'

    def __init__(self, *args, **kwargs):
        super(FileWriter, self).__init__(*args, **kwargs)

        # See if we should compress output files or not
        self._compress = self._cfg.getbool('soln-output', 'compress', False)

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the type and shape of each element in the partition
        types, shapes = self._meshp.ele_types, self._meshp.ele_shapes

        # Gather this information onto the root rank
        eleinfo = comm.gather(zip(types, shapes), root=root)

        if rank == root:
            self._mpi_rbufs = mpi_rbufs = []
            self._mpi_rreqs = mpi_rreqs = []
            self._mpi_names = mpi_names = []
            self._loc_names = loc_names = []

            for mrank, meleinfo in enumerate(eleinfo):
                prank = self._rallocs.mprankmap[mrank]
                for tag, (type, dims) in enumerate(meleinfo):
                    name = self._get_name_for_soln(type, prank)

                    if mrank == root:
                        loc_names.append(name)
                    else:
                        rbuf = np.empty(dims, dtype=self._backend.fpdtype)
                        rreq = comm.Recv_init(rbuf, mrank, tag)

                        mpi_rbufs.append(rbuf)
                        mpi_rreqs.append(rreq)
                        mpi_names.append(name)

    def _write(self, path, solnmap, metadata):
        comm, rank, root = get_comm_rank_root()

        if rank != root:
            for tag, buf in enumerate(solnmap.values()):
                comm.Send(buf.copy(), root, tag)
        else:
            # Recv all of the non-local solution mats
            MPI.Prequest.Startall(self._mpi_rreqs)
            MPI.Prequest.Waitall(self._mpi_rreqs)

            # Combine local and MPI data
            names = itertools.chain(self._loc_names, self._mpi_names)
            solns = itertools.chain(solnmap.values(), self._mpi_rbufs)

            # Create the output dictionary
            outdict = dict(zip(names, solns), **metadata)

            with open(path, 'wb') as f:
                if self._compress:
                    np.savez_compressed(f, **outdict)
                else:
                    np.savez(f, **outdict)

class DirWriter(BaseWriter):
    writer_name = 'pyfrs-dir'

    def _write(self, path, solnmap, metadata):
        comm, rank, root = get_comm_rank_root()

        # Create the output directory and save the config/status files
        if rank == root:
            if os.path.exists(path):
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)

            os.mkdir(path)

            # Write out our metadata
            for name, data in metadata.items():
                np.save(os.path.join(path, name), data)

        # Wait for this to complete
        comm.barrier()

        # Save the solutions
        for type, buf in solnmap.items():
            solnpath = os.path.join(path, self._get_name_for_soln(type))
            np.save(solnpath, buf)
