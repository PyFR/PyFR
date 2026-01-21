from pyfr.mpiutil import get_comm_rank_root, mpi


class Serialiser:
    def __init__(self):
        self._serialfns = {}

    def register(self, prefix, datafn):
        comm, rank, root = get_comm_rank_root()
        sender = comm.exscan(1 if datafn else 0) == 0 and datafn

        # If root doesn't have the data, lowest rank with it sends it to root
        if rank == root and not datafn:
            sendrank = comm.recv()
            sfn = lambda: comm.recv(source=sendrank)
        elif sender:
            comm.send(rank, root)
            sfn = lambda: comm.send(datafn(), root)
        else:
            sfn = datafn
        
        if sfn:
            self._serialfns[prefix] = sfn
    
    def serialise(self):
        return {k: sfn() for k, sfn in self._serialfns.items()}
