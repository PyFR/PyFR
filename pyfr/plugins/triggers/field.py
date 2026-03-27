from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
from pyfr.plugins.fieldeval import BackendFieldReducer
from pyfr.plugins.triggers.base import BaseTriggerSource


_red_map = {
    'min': ('min', mpi.MIN),
    'max': ('max', mpi.MAX),
    'sum': ('sum', mpi.SUM),
    'avg': ('sum', mpi.SUM),
    'l2norm': ('sum', mpi.SUM),
}


class FieldTriggerSource(BaseTriggerSource):
    name = 'field'
    collective = True

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)

        self._nsteps = cfg.getint(cfgsect, 'nsteps')

        red, expr, cmp, threshold = self._parse_condition(cfg, cfgsect,
                                                          _red_map)
        self._red_name = red
        reduceop, self._mpi_op = _red_map[red]
        self._cmp = cmp
        self._threshold = threshold

        # Wrap l2norm expressions
        if red == 'l2norm':
            expr = f'({expr})*({expr})'

        self._freduce = BackendFieldReducer(intg.backend, cfg, cfgsect, intg,
                                            [expr], reduceop)
        self._last_result = False

        # Precompute the global total volume for avg reductions
        if red == 'avg':
            comm, _, _ = get_comm_rank_root()
            local_vol = self._freduce.total_volume()
            self._total_vol = scal_coll(comm.Allreduce, local_vol)

    def evaluate(self, intg):
        if intg.nacptsteps % self._nsteps != 0:
            return self._last_result

        comm, rank, root = get_comm_rank_root()

        local_val = self._freduce(intg)[0]

        if self._red_name == 'avg':
            gsum = scal_coll(comm.Allreduce, local_val)
            val = gsum / self._total_vol if self._total_vol > 0 else 0.0
        elif self._red_name == 'l2norm':
            val = scal_coll(comm.Allreduce, local_val, op=self._mpi_op)**0.5
        else:
            val = scal_coll(comm.Allreduce, local_val, op=self._mpi_op)

        self._last_result = self._cmp(val, self._threshold)
        return self._last_result
