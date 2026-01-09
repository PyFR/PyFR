import itertools as it

from pyfr.mpiutil import get_comm_rank_root, mpi

class Serialiser:
    BASE_MPI_TAG = 9000

    def __init__(self):
        self._serialfns = {}
        self._mpi_tag_counter = it.count(self.BASE_MPI_TAG)

    @classmethod
    def serialisefn(cls, datafn, tag):
        comm, rank, root = get_comm_rank_root()
        rootdata = rank == root and datafn
        sender = comm.exscan(1 if datafn else 0) == 0 and datafn

        if rootdata:
            return datafn
        else:
            if rank == root:
                return lambda: comm.recv(buf=None, source=mpi.ANY_SOURCE, tag=tag)
            elif sender:
                def _send():
                    comm.send(datafn(), root, tag)
                    return {}
                return _send
            else:
                return None

    def register_sdata(self, prefix, datafn):
        mpi_tag = next(self._mpi_tag_counter)
        sfn = Serialiser.serialisefn(datafn, mpi_tag)
        if sfn is not None:
            self._serialfns[prefix] = sfn
    
    def serialise(self):
        return {f'sdata/{k}': sfn() for k, sfn in self._serialfns.items()}
