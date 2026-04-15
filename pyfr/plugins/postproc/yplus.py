import numpy as np

from pyfr.plugins.postproc.base import BasePostProcPlugin


class YPlusPostProc(BasePostProcPlugin):
    name = 'yplus'
    systems = 'navier-stokes'
    dimensions = '3'
    export_types = 'boundary'
    needs_grads = True

    def fields(self):
        return {'yplus': ['y+']}

    def process(self, data):
        mu = self.cfg.getfloat('constants', 'mu')
        rho_wall = data.pris[0]

        u_tau = np.sqrt(data.tau_wall / rho_wall)
        nu = mu / rho_wall

        data.fields['yplus'] = data.min_upt_wall_dist_approx * u_tau / nu
