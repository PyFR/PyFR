from pyfr.mpiutil import get_comm_rank_root

class Serialiser:

    @classmethod
    def serialisefn(cls, datafn):
        comm, rank, root = get_comm_rank_root()
        rootHasData = datafn is not None
        rootHasData = comm.bcast(rootHasData, root=root)

        if rootHasData:
            if rank == root:
                return datafn
            else:
                return None
        else:
            isSender = comm.exscan(1 if datafn else 0) == 0 and datafn
            if rank == root:
                return comm.recv
            elif isSender:
                def _send():
                    comm.send(datafn(), root)
                    return {}
                return _send
            else:
                return None