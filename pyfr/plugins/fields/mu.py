import numpy as np

from pyfr.plugins.fields.base import BaseFieldProvider


class MuField(BaseFieldProvider):
    name = '_mu'
    systems = 'navier-stokes'
    dimensions = '2|3'
    export_types = '.*'

    def _process(self, view):
        cfg = view.cfg
        mu_ref = cfg.getfloat('constants', 'mu')

        if cfg.get('solver', 'viscosity-correction', 'none') == 'sutherland':
            gamma = cfg.getfloat('constants', 'gamma')
            cpTref = cfg.getfloat('constants', 'cpTref')
            cpTs = cfg.getfloat('constants', 'cpTs')

            rho, p = view.pris[0], view.pris[-1]
            cpT = gamma * p / ((gamma - 1) * rho)
            Trat = cpT / cpTref
            view.fields['_mu'] = (mu_ref * (cpTref + cpTs) * Trat
                                  * np.sqrt(Trat) / (cpT + cpTs))
        else:
            view.fields['_mu'] = mu_ref
