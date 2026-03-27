import numpy as np

from pyfr.plugins.postproc.base import BasePostProcPlugin


class MachPostProc(BasePostProcPlugin):
    name = 'mach'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'
    export_types = '.*'

    def fields(self):
        return {'mach': ['Ma']}

    def process(self, data):
        rho, *vs, p = data.pris

        gamma = self.cfg.getfloat('constants', 'gamma')
        vmag = np.sqrt(sum(v**2 for v in vs))
        c = np.sqrt(gamma * p / rho)

        data.fields['mach'] = [vmag / c]
