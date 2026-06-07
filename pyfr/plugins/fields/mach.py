import numpy as np

from pyfr.plugins.fields.base import BaseFieldProvider


class MachField(BaseFieldProvider):
    name = 'mach'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'
    export_types = '.*'
    fields = {'mach': ['Ma']}

    def _process(self, view):
        rho, *vs, p = view.pris

        gamma = self.cfg.getfloat('constants', 'gamma')
        vmag = np.sqrt(sum(v**2 for v in vs))
        c = np.sqrt(gamma * p / rho)

        view.fields['mach'] = vmag / c
