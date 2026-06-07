from pyfr.plugins.fields.base import BaseFieldProvider


class CpField(BaseFieldProvider):
    name = 'cp'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'
    export_types = '.*'
    fields = {'cp': ['Cp']}

    def _process(self, view):
        p = view.pris[-1]

        rho_inf = self.cfg.getfloat('constants', 'rho-inf')
        u_inf = self.cfg.getfloat('constants', 'u-inf')
        p_inf = self.cfg.getfloat('constants', 'p-inf')

        q_inf = 0.5 * rho_inf * u_inf**2

        view.fields['cp'] = (p - p_inf) / q_inf
