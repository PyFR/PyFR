from pyfr.plugins.postproc.base import BasePostProcPlugin


class CfPostProc(BasePostProcPlugin):
    name = 'cf'
    systems = 'navier-stokes'
    dimensions = '3'
    export_types = 'boundary'
    deps = ['_tau_wall']

    def fields(self):
        return {'cf': ['Cf']}

    def _process(self, data):
        rho_inf = self.cfg.getfloat(self.cfgsect, 'rho-inf')
        u_inf = self.cfg.getfloat(self.cfgsect, 'u-inf')

        q_inf = 0.5 * rho_inf * u_inf**2

        data.fields['cf'] = data.fields['_tau_wall'] / q_inf
