import math

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
from pyfr.plugins.soln.base import BaseSolnPlugin


class NaNCheckPlugin(BaseSolnPlugin):
    name = 'nancheck'
    systems = '.*'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self._backend = intg.backend
        self._ele_banks = intg.system.ele_banks

    @memoize
    def _sum_kerns(self, idx):
        return [self._backend.kernel('reduction', 'sum', ['x'], {'x': eb[idx]})
                for eb in self._ele_banks]

    def __call__(self, intg):
        kerns = self._sum_kerns(intg.idxcurr)
        self._backend.run_kernels(kerns, wait=True)

        has_nan = any(math.isnan(k.retval[0]) for k in kerns)

        # Sync across ranks if we have a trigger to fire
        if self.trigger_fire_name:
            comm, _, _ = get_comm_rank_root()
            has_nan = scal_coll(comm.Allreduce, int(has_nan), op=mpi.LOR)

        if has_nan:
            if self.trigger_fire_name:
                intg.fire_trigger(self.trigger_fire_name)

            intg.plugin_abort(f'NaNs detected at t = {intg.tcurr}')
