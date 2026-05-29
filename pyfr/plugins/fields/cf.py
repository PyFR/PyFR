from pyfr.plugins.fields.base import BaseFieldProvider


class CfField(BaseFieldProvider):
    name = 'cf'
    systems = 'navier-stokes'
    dimensions = '3'
    export_types = 'boundary'
    deps = ['_tau_wall']
    fields = {'cf': ['Cf']}

    def _process(self, view):
        rho_inf = self.cfg.getfloat('constants', 'rho-inf')
        u_inf = self.cfg.getfloat('constants', 'u-inf')

        q_inf = 0.5 * rho_inf * u_inf**2

        view.fields['cf'] = view.fields['_tau_wall'] / q_inf
