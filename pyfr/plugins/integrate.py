from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import (BackendMixin, BaseSolnPlugin, PublishMixin,
                               init_csv)
from pyfr.plugins.fieldeval import BackendFieldReducer


class IntegratePlugin(PublishMixin, BackendMixin, BaseSolnPlugin):
    name = 'integrate'
    systems = ['*']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self._init_backend(intg)
        comm, rank, root = get_comm_rank_root()

        # Expressions to integrate
        c = self.cfg.items_as('constants', float)
        self._inames = self.cfg.items(cfgsect, prefix='int-')
        self.exprs = [self.cfg.getexpr(cfgsect, k, subs=c)
                      for k in self._inames]

        # Add optional integral L-p degree
        if self.cfg.get(cfgsect, 'norm', 'none').lower() != 'none':
            self.lp = self.cfg.getfloat(cfgsect, 'norm')

            # Modify expressions for L-p norm
            if self.lp == 1 or self.lp == float('inf'):
                self.exprs = [f'abs({e})' for e in self.exprs]
                self._post_func = lambda x: x
            else:
                self.exprs = [f'pow(abs({e}), {self.lp})' for e in self.exprs]
                self._post_func = lambda x: x**(1 / self.lp)
        else:
            self.lp = None
            self._post_func = lambda x: x

        # Set MPI reduction op and backend reduction op
        if self.lp == float('inf'):
            self.mpi_op = mpi.MAX
            reduceop = 'max'
        else:
            self.mpi_op = mpi.SUM
            reduceop = 'sum'

        # Backend field evaluation and reduction
        self._freduce = BackendFieldReducer(self.backend, self.cfg, cfgsect,
                                            intg, self.exprs, reduceop)

        # The root rank needs to open the output file
        if rank == root:
            header = ['t', *self.cfg.items(cfgsect, prefix='int-')]

            # Open
            self.csv = init_csv(self.cfg, cfgsect, ','.join(header), nflush=1)

    def __call__(self, intg):
        comm, rank, root = get_comm_rank_root()

        # Evaluate expressions on the backend
        iintex = self._freduce(intg)

        # Reduce and output if we're the root rank
        if rank != root:
            comm.Reduce(iintex, None, op=self.mpi_op, root=root)
        else:
            comm.Reduce(mpi.IN_PLACE, iintex, op=self.mpi_op, root=root)

            # Apply post integration function
            iintex = [self._post_func(i) for i in iintex]

            # Write
            self.csv(intg.tcurr, *iintex)

            # Publish integral values
            self._publish(intg, **dict(zip(self._inames, iintex)))
