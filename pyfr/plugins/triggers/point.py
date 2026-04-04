import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.nputil import npeval
from pyfr.plugins.triggers.base import BaseTriggerSource
from pyfr.points import PointSampler
from pyfr.util import first


class PointTriggerSource(BaseTriggerSource):
    name = 'point'
    collective = True

    _valid_reds = {'min', 'max', 'avg'}

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)

        self._nsteps = cfg.getint(cfgsect, 'nsteps')

        red, expr, cmp, threshold = self._parse_condition(
            cfg, cfgsect, self._valid_reds
        )
        self._red_name = red
        self._expr = expr
        self._cmp = cmp
        self._threshold = threshold

        system = intg.system
        self._elementscls = system.elementscls
        self._privars = first(system.ele_map.values()).privars

        self._psampler = PointSampler(system.mesh,
                                      cfg.getliteral(cfgsect, 'pts'))
        self._psampler.configure_with_intg_nvars(intg, system.nvars)
        self._last_result = False

    def evaluate(self, intg):
        if intg.nacptsteps % self._nsteps != 0:
            return self._last_result

        # All ranks participate in sampling and gathering
        samps = self._psampler.sample(intg.soln)

        comm, rank, root = get_comm_rank_root()

        if rank == root:
            # Convert conservative to primitive
            psamps = self._elementscls.con_to_pri(samps.T, self.cfg)
            subs = dict(zip(self._privars, psamps))
            vals = npeval(self._expr, subs)

            match self._red_name:
                case 'min':
                    val = float(np.amin(vals))
                case 'max':
                    val = float(np.amax(vals))
                case 'avg':
                    val = float(np.mean(vals))

            self._last_result = self._cmp(val, self._threshold)

        self._last_result = comm.bcast(self._last_result, root=root)
        return self._last_result
