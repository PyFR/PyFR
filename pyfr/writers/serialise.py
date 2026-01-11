from pyfr.mpiutil import get_comm_rank_root, mpi


class Serialiser:
    def __init__(self):
        self._serialfns = {}

    def serialisefn(self, datafn):
        comm, rank, root = get_comm_rank_root()
        sender = comm.exscan(1 if datafn else 0) == 0 and datafn

        fn = datafn
        if rank == root and not datafn:
            sendrank = comm.recv()
            fn = lambda: comm.recv(source=sendrank)
        elif sender:
            comm.send(rank, root)
            fn = lambda: comm.send(datafn(), root)
        
        return fn

    def register_sdata(self, prefix, datafn):
        sfn = self.serialisefn(datafn)
        if sfn is not None:
            self._serialfns[prefix] = sfn
    
    def serialise(self):
        return {k: sfn() for k, sfn in self._serialfns.items()}
