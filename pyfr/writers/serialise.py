from pyfr.mpiutil import get_comm_rank_root, mpi


class Serialiser:
    def __init__(self):
        self._serialfns = {}

    def serialisefn(self, datafn):
        comm, rank, root = get_comm_rank_root()
        sender = comm.exscan(1 if datafn else 0) == 0 and datafn

        if rank == root and not datafn:
            sendrank = comm.recv()
            return lambda: comm.recv(source=sendrank)
        elif sender:
            comm.send(rank, root)
            return lambda: comm.send(datafn(), root)
        else:
            return datafn

    def register_sdata(self, prefix, datafn):
        if (sfn := self.serialisefn(datafn)):
            self._serialfns[prefix] = sfn
    
    def serialise(self):
        return {k: sfn() for k, sfn in self._serialfns.items()}
