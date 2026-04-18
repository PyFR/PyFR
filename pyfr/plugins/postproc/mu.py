import numpy as np

from pyfr.plugins.postproc.base import BasePostProcPlugin


class MuPostProc(BasePostProcPlugin):
    name = '_mu'
    systems = 'navier-stokes'
    dimensions = '2|3'
    export_types = '.*'

    def fields(self):
        return {}

    def _process(self, data):
        cfg = data.cfg
        mu_ref = cfg.getfloat('constants', 'mu')

        if cfg.get('solver', 'viscosity-correction', 'none') == 'sutherland':
            gamma = cfg.getfloat('constants', 'gamma')
            cpTref = cfg.getfloat('constants', 'cpTref')
            cpTs = cfg.getfloat('constants', 'cpTs')

            rho, p = data.pris[0], data.pris[-1]
            cpT = gamma * p / ((gamma - 1) * rho)
            Trat = cpT / cpTref
            data.fields['_mu'] = (mu_ref * (cpTref + cpTs) * Trat
                                  * np.sqrt(Trat) / (cpT + cpTs))
        else:
            data.fields['_mu'] = mu_ref
