# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import re

from pyfr.mpiutil import get_comm_rank_root
from pyfr.util import subclass_where


def get_rank_allocation(mesh, cfg):
    name = cfg.get('backend', 'rank-allocator', 'linear')

    return subclass_where(BaseRankAllocator, name=name)(mesh, cfg)


class BaseRankAllocator(object, metaclass=ABCMeta):
    name = None

    def __init__(self, mesh, cfg):
        self.cfg = cfg

        comm, rank, root = get_comm_rank_root()

        if rank == root:
            # Determine the (physical) connectivity of the mesh
            prankconn = self._get_mesh_connectivity(mesh)
            nparts = len(prankconn) or 1

            if nparts != comm.size:
                raise RuntimeError('Mesh has %d partitions but running with '
                                   '%d MPI ranks' % (nparts, comm.size))
        else:
            prankconn = None

        # Get subclass dependant info about each rank (e.g, hostname)
        rinfo = comm.gather(self._get_rank_info(), root=root)

        if rank == root:
            # Use this info to construct a mapping from MPI ranks to
            # physical mesh ranks
            mprankmap = self._get_mprankmap(prankconn, rinfo)
        else:
            mprankmap = None

        # Broadcast the connectivity and physical to each MPI rank
        self.prankconn = comm.bcast(prankconn, root=root)
        self.mprankmap = comm.bcast(mprankmap, root=root)

        # Invert this mapping
        self.pmrankmap = {v: k for k, v in self.mprankmap.items()}

        # Compute the physical rank of ourself
        self.prank = self.mprankmap[rank]

    def _get_mesh_connectivity(self, mesh):
        conn = defaultdict(list)
        for f in mesh:
            m = re.match(r'con_p(\d+)p(\d+)$', f)
            if m:
                lhs, rhs = int(m.group(1)), int(m.group(2))
                conn[lhs].append(rhs)

                if 'con_p%dp%d' % (rhs, lhs) not in mesh:
                    raise ValueError('MPI interface (%d, %d) is not symmetric'
                                     % (lhs, rhs))

        if sorted(conn) != list(range(len(conn))):
            raise ValueError('Mesh has invalid partition numbers')

        return conn

    @abstractmethod
    def _get_rank_info(self):
        pass

    @abstractmethod
    def _get_mprankmap(self, prankconn, rinfo):
        pass


class LinearRankAllocator(BaseRankAllocator):
    name = 'linear'

    def _get_rank_info(self):
        return None

    def _get_mprankmap(self, prankconn, rinfo):
        return {i: i for i in range(len(rinfo))}
