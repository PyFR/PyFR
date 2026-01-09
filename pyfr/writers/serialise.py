from pyfr.mpiutil import get_comm_rank_root, mpi

class Serialiser:
    def __init__(self):
        self._serialfns = {}

    @classmethod
    def serialisefn(cls, datafn):
        comm, rank, root = get_comm_rank_root()
        rootdata = rank == root and datafn
        sender = comm.exscan(1 if datafn else 0) == 0 and datafn

        if rootdata:
            return datafn
        else:
            if rank == root:
                sendrank = comm.recv()
                return lambda: comm.recv(source=sendrank)
            elif sender:
                comm.send(rank, root)
                def _send():
                    comm.send(datafn(), root)
                    return {}
                return _send
            else:
                return None

    def register_sdata(self, prefix, datafn):
        sfn = Serialiser.serialisefn(datafn)
        if sfn is not None:
            self._serialfns[prefix] = sfn
    
    def serialise(self):
        return {f'sdata/{k}': sfn() for k, sfn in self._serialfns.items()}
