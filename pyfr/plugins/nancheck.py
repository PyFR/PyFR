import numpy as np

from pyfr.plugins.base import BaseSolnPlugin


class NaNCheckPlugin(BaseSolnPlugin):
    name = 'nancheck'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            if any(np.isnan(np.sum(s)) for s in intg.soln):
                intg.plugin_abort(f'NaNs detected at t = {intg.tcurr}')
