# -*- coding: utf-8 -*-

import re

from abc import ABCMeta, abstractmethod
from collections import defaultdict

from mpi4py import MPI

from pyfr.util import all_subclasses


def get_rank_allocation(mesh, cfg):
    allocators = {a.name: a for a in all_subclasses(BaseRankAllocator)}
    return allocators[cfg.get('mesh', 'allocator')](mesh, cfg)

class BaseRankAllocator(object):
    __metaclass__ = ABCMeta

    name = None

    def __init__(self, mesh, cfg):
        self._cfg = cfg

        comm = MPI.COMM_WORLD
        size = comm.size
        rank = comm.rank
        root = 0

        if rank == root:
            # Determine the (physical) connectivity of the mesh
            prankconn = self._get_mesh_connectivity(mesh)
            nparts = len(prankconn) or 1

            if nparts != size:
                raise RuntimeError('Mesh has %d partitions but running with '
                                   '%d MPI ranks' % (nparts, size))
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
        self.pmrankmap = {v: k for k,v in self.mprankmap.items()}

    def _get_mesh_connectivity(self, mesh):
        conn = defaultdict(list)
        for f in mesh:
            m = re.match('con_p(\d+)p(\d+)', f)
            if m:
                lhs, rhs = int(m.group(1)), int(m.group(2))
                conn[lhs].append(rhs)

                if 'con_p%dp%d' % (rhs, lhs) not in mesh:
                    raise ValueError('MPI interface (%d, %d) is not symmetric'
                                   % (lhs, rhs))

        if list(sorted(conn.keys())) != list(range(len(conn))):
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
        return {i: i for i in xrange(len(rinfo))}
