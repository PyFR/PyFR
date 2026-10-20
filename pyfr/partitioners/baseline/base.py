from pyfr.partitioners.base import BasePartitioner
from pyfr.partitioners.baseline.multilevel import GraphPartitioner


class BaselinePartitioner(BasePartitioner):
    name = 'baseline'
    has_part_weights = True
    has_multiple_constraints = True

    int_opts = frozenset({
        'seed', 'nrefine', 'ufactor', 'niter', 'coarsen_min_mult',
        'coarsen_min_abs', 'coarsen_stop_pct'
    })
    enum_opts = {'ptype': {'kway': 'kway', 'rb': 'rb'}}
    dflt_opts = {
        'ptype': 'kway',
        'seed': 2079, 'nrefine': 10, 'ufactor': 10, 'niter': 1,
        'coarsen_min_mult': 20, 'coarsen_min_abs': 500,
        'coarsen_stop_pct': 80
    }

    def _partition_graph(self, graph, partwts):
        opts = self.opts
        gp = GraphPartitioner(
            seed=opts['seed'], nrefine=opts['nrefine'],
            nstarts=max(4, opts['niter']), imbalance=opts['ufactor'] / 1000,
            coarsen_min_mult=opts['coarsen_min_mult'],
            coarsen_min_abs=opts['coarsen_min_abs'],
            coarsen_stop=opts['coarsen_stop_pct'] / 100
        )

        return gp.partition(graph, partwts, opts['ptype'])
