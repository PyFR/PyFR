from pyfr.plugins.postproc.base import BasePostProcPlugin


class VorticityPostProc(BasePostProcPlugin):
    name = 'vorticity'
    systems = '.*'
    dimensions = '2|3'
    export_types = '.*'
    needs_grads = True

    def fields(self):
        if self.ndims == 3:
            return {'vorticity': ['omega_x', 'omega_y', 'omega_z']}
        else:
            return {'vorticity': ['omega_z']}

    def process(self, data):
        du = data.grad_pris[1]
        dv = data.grad_pris[2]

        if self.ndims == 3:
            dw = data.grad_pris[3]

            omega_x = dw[1] - dv[2]
            omega_y = du[2] - dw[0]
            omega_z = dv[0] - du[1]

            data.fields['vorticity'] = [omega_x, omega_y, omega_z]
        else:
            omega_z = dv[0] - du[1]

            data.fields['vorticity'] = [omega_z]
