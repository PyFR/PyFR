import numpy as np

from pyfr.plugins.postproc.base import BasePostProcPlugin


class IsentropicMachPostProc(BasePostProcPlugin):
    name = 'isen-mach'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'
    export_types = '.*'

    def fields(self):
        return {'isen-mach': ['Ma_is']}

    def process(self, data):
        p = data.pris[-1]

        gamma = self.cfg.getfloat('constants', 'gamma')
        p_t = self.cfg.getfloat(self.cfgsect, 'p-total')

        gm1 = gamma - 1
        data.fields['isen-mach'] = np.sqrt(2/gm1*((p_t/p)**(gm1/gamma) - 1))
