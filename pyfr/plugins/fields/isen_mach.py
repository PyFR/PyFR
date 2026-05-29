import numpy as np

from pyfr.plugins.fields.base import BaseFieldProvider


class IsentropicMachField(BaseFieldProvider):
    name = 'isen-mach'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'
    export_types = '.*'
    fields = {'isen-mach': ['Ma_is']}

    def _process(self, view):
        p = view.pris[-1]

        gamma = self.cfg.getfloat('constants', 'gamma')
        p_t = self.cfg.getfloat('constants', 'p-total')

        gm1 = gamma - 1
        view.fields['isen-mach'] = np.sqrt(2/gm1*((p_t/p)**(gm1/gamma) - 1))
