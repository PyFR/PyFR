import os
from pyfr.plugins.base import BaseSolnPlugin

class StopPlugin(BaseSolnPlugin):
    name = 'stop'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        # Check for this file to trigger a graceful stop
        self.stop_file = self.cfg.get(cfgsect, 'file', 'STOP')

    def __call__(self, intg):
        if os.path.exists(self.stop_file):
            # The root rank handles the notification and save triggering
            from pyfr.mpiutil import get_comm_rank_root
            comm, rank, root = get_comm_rank_root()
            
            if rank == root:
                print(f"Graceful stop requested via {self.stop_file}")
                
            # Try to trigger a final save if a writer exists
            for p in intg.plugins:
                if getattr(p, 'name', None) == 'writer':
                    # Force the writer to output now
                    p.tout_last = intg.tcurr - p.dt_out - 1e-12
                    p(intg)
                    break
            
            # Request abort from the integrator
            intg.plugin_abort(f"Graceful stop requested via {self.stop_file}")
            
            # Clean up the abort file on root
            if rank == root:
                try:
                    os.remove(self.stop_file)
                except Exception:
                    pass
