import numpy as np

from pyfr.plugins.fields.base import BaseFieldProvider


class VorticityField(BaseFieldProvider):
    name = 'vorticity'
    systems = '.*'
    dimensions = '2|3'
    export_types = '.*'
    needs_grads = True

    @property
    def fields(self):
        if self.ndims == 3:
            return {'vorticity': ('omega_x', 'omega_y', 'omega_z')}
        else:
            return {'vorticity': ('omega_z',)}

    def _process(self, view):
        du = view.grad_pris[1]
        dv = view.grad_pris[2]

        if self.ndims == 3:
            dw = view.grad_pris[3]

            omega_x = dw[1] - dv[2]
            omega_y = du[2] - dw[0]
            omega_z = dv[0] - du[1]

            view.fields['vorticity'] = np.stack(
                [omega_x, omega_y, omega_z], axis=-1
            )
        else:
            view.fields['vorticity'] = dv[0] - du[1]
