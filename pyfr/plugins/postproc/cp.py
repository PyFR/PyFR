from pyfr.plugins.postproc.base import BasePostProcPlugin


class CpPostProc(BasePostProcPlugin):
    name = 'cp'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'
    export_types = '.*'

    def fields(self):
        return {'cp': ['Cp']}

    def _process(self, data):
        p = data.pris[-1]

        rho_inf = self.cfg.getfloat(self.cfgsect, 'rho-inf')
        u_inf = self.cfg.getfloat(self.cfgsect, 'u-inf')
        p_inf = self.cfg.getfloat(self.cfgsect, 'p-inf')

        q_inf = 0.5 * rho_inf * u_inf**2

        data.fields['cp'] = (p - p_inf) / q_inf
