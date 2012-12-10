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
        self._nout = 0

    def output(self, solnmap, stats):
        comm, rank, root = get_comm_rank_root()

        # Convert the config and stats objects to strings
        if rank == root:
            cfg_s = self._cfg.tostr()
            stats_s = stats.tostr()
        else:
            cfg_s = None
            stats_s = None

        # Determine the output path
        path = self._get_output_path()

        # Delegate to _write to do the actual outputting
        self._write(path, solnmap, cfg_s, stats_s)

        # Increment the output number
        self._nout += 1

    @abstractmethod
    def _write(self, path, solnmap, cfg_s, stat_s):
        pass

    def _get_output_path(self):
        # Get the output directory
        d = self._basedir

        # Current time and output number
        t = format(self._tcurr)
        n = format(self._nout)

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
                        rbuf = np.empty(dims)
                        rreq = comm.Recv_init(rbuf, mrank, tag)

                        mpi_rbufs.append(rbuf)
                        mpi_rreqs.append(rreq)
                        mpi_names.append(name)

    def _write(self, path, solnmap, cfg_s, stats_s):
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

            with open(path, 'wb') as f:
                np.savez(f, config=cfg_s, stats=stats_s,
                         **dict(zip(names, solns)))


class DirWriter(BaseWriter):
    writer_name = 'pyfrs-dir'

    def _write(self, path, solnmap, cfg_s, stats_s):
        comm, rank, root = get_comm_rank_root()

        # Create the output directory and save the config/status files
        if rank == root:
            if os.path.exists(path):
                shutil.rmtree(path)

            os.mkdir(path)

            with open(os.path.join(path, 'config'), 'wb') as f:
                f.write(np.asanyarray(cfg_s))
            with open(os.path.join(path, 'stats'), 'wb') as f:
                f.write(np.asanyarray(stats_s))

        # Wait for this to complete
        comm.barrier()

        # Save the solutions
        for type, buf in solnmap.items():
            solnpath = os.path.join(path, self._get_name_for_soln(type))
            np.save(solnpath, buf)
