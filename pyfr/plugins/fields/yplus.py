import numpy as np

from pyfr.plugins.fields.base import BaseFieldProvider


class YPlusField(BaseFieldProvider):
    name = 'yplus'
    systems = 'navier-stokes'
    dimensions = '3'
    export_types = 'boundary'
    deps = ['_tau_wall', '_mu']
    fields = {'yplus': ['y+']}

    def _process(self, view):
        rho_wall = view.pris[0]

        u_tau = np.sqrt(view.fields['_tau_wall'] / rho_wall)
        nu = view.fields['_mu'] / rho_wall

        view.fields['yplus'] = view.min_upt_wall_dist_approx * u_tau / nu
