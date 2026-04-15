import numpy as np

from pyfr.plugins.postproc.base import BasePostProcPlugin


class YPlusPostProc(BasePostProcPlugin):
    name = 'yplus'
    systems = 'navier-stokes'
    dimensions = '3'
    export_types = 'boundary'
    deps = ['_tau_wall', '_mu']

    def fields(self):
        return {'yplus': ['y+']}

    def _process(self, data):
        rho_wall = data.pris[0]

        u_tau = np.sqrt(data.fields['_tau_wall'] / rho_wall)
        nu = data.fields['_mu'] / rho_wall

        data.fields['yplus'] = data.min_upt_wall_dist_approx * u_tau / nu
